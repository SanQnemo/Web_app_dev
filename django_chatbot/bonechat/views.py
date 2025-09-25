from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator

from django.contrib import auth
from django.contrib.auth.models import User

from django.db import IntegrityError
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from django.conf import settings
import logging

logger = logging.getLogger(__name__)
from django.contrib.auth import get_user_model

import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64
import json
from dotenv import load_dotenv
import logging

from .models import Chat
from django.utils import timezone  

from bonechat.models import Chat
from django.contrib.auth.decorators import login_required

from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tempfile
from pathlib import Path

# ===== Load ENV & Configure =====
load_dotenv()
API = os.getenv("LMS_API")
URL = os.getenv("base_url_lms")
embed_model = "google/embeddinggemma-300m"

faiss_storage = settings.BASE_DIR / "faiss_storage"
faiss_text = faiss_storage / "faiss_text" 
faiss_tabular = faiss_storage / "faiss_tabular"

faiss_text.mkdir(parents=True, exist_ok=True)
faiss_tabular.mkdir(parents=True, exist_ok=True)

system_template_dir = settings.SYSTEM_IT_PATH  
system_template_file = settings.SYSTEM_TEMPLATE_FILE  

system_template_dir.mkdir(parents=True, exist_ok=True)

print("Created directories:")
print(faiss_text)
print(faiss_tabular)
print(system_template_dir)
print(system_template_file)

# ใช้บริการบน LM studio
client = OpenAI(base_url = f"{URL.rstrip('/')}/v1",api_key=API)

# โหลด embedder
embedder = HuggingFaceEmbeddings(
    model_name = embed_model,
    encode_kwargs = {"normalize_embeddings": True}
)

# โหลด vector database
# Load FAISS indices (load once at startup)
try:
    index_text = FAISS.load_local(faiss_text, embedder, allow_dangerous_deserialization=True)
    index_tabular = FAISS.load_local(faiss_tabular, embedder, allow_dangerous_deserialization=True)
    FAISS_LOADED = True
except Exception as e:
    logger.error(f"Failed to load FAISS indices: {e}")
    FAISS_LOADED = False

# ===== Image Handling Utilities =====
def handle_uploaded_images(uploaded_files):
    """Save uploaded images to temporary files and return their paths"""
    temp_paths = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
                # Write the uploaded file content to temporary file
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_paths.append(temp_file.name)
                logger.info(f"Saved uploaded image to: {temp_file.name}")
        except Exception as e:
            logger.error(f"Error saving uploaded file {uploaded_file.name}: {e}")
    
    return temp_paths

def cleanup_temp_files(file_paths):
    """Clean up temporary files after processing"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None
    
def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None
    
# ===== RAG Utility Functions =====
def load_system_template(template_path, is_medical_analysis):
    """Load system template - only use strict format for medical image analysis"""
    try:
        with open(template_path, "r", encoding='utf-8') as f:
            json_template = json.load(f)
        
        if is_medical_analysis:
            # Strict format only for medical image analysis
            system_prompt = (
                "You are a Radiology Expert. You will be given X-ray images, analyze them and always answer "
                "in the following structured format and use based knowledge that provided from retrieval within 512 tokens.\n"
                "Do not reference which sources used,just answer based on format given"
    
                "Answer format:\n"
                + json.dumps(json_template, indent=2)
            )
        else:
            # Flexible format for general medical questions
            system_prompt = (
                "You are a helpful Medical AI Assistant that specialized in radiology. Provide clear, conversational answers to medical questions. "
                "For general questions, answer naturally without forced formatting. "
                "Only use structured medical report format when analyzing actual X-ray images."
            )
        
        return system_prompt
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        return "You are a helpful Medical AI Assistant. Provide clear, informative answers."

def get_rag_context(query_prompt):
    """Retrieve relevant context only for medical queries"""
    # Simple greetings and non-medical queries should not use RAG
    non_medical_keywords = ['hello', 'hi', 'hey', 'how are you', 'thanks', 'thank you', 'goodbye', 'bye']
    
    if any(keyword in query_prompt.lower() for keyword in non_medical_keywords):
        return "General conversation - no medical context needed."
    
    if not FAISS_LOADED:
        return "RAG system not available."
    
    try:
        # Only query if it looks like a medical question
        medical_keywords = ['x-ray', 'fracture', 'bone', 'medical', 'diagnose', 'RH', 'RI', 'UV','report']
        if not any(keyword in query_prompt.lower() for keyword in medical_keywords):
            return "General medical question - using standard knowledge."
        
        # Query both indices
        query_results_csv = index_tabular.similarity_search(query_prompt, k=3)  # Reduced from 3
        query_results_pdf = index_text.similarity_search(query_prompt, k=3)     # Reduced from 3
        
        # Combine results
        combined_results = []
        
        for doc in query_results_csv:
            combined_results.append(f"Clinical Data: {doc.page_content}")
        
        for doc in query_results_pdf:
            combined_results.append(f"Medical Literature: {doc.page_content[:300]}...")  # Reduced length
        
        return "\n".join(combined_results) if combined_results else "No relevant medical context found."
    
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        return "Error retrieving medical context."
    
def generate_radiology_report(user_message, image_paths=None, clinical_data=None):
    """Generate radiology report using RAG and image analysis"""
    temp_files_to_cleanup = []
    
    try:
         # Determine if this is actual medical image analysis
        has_images = image_paths and len(image_paths) > 0
        is_medical_analysis = has_images or any(keyword in user_message.lower() for keyword in [
            "diagnose","report"
        ])
        
        # Load appropriate template
        system_instruction = load_system_template(system_template_file, is_medical_analysis)
        
        # Get RAG context and incorporate it into the system prompt
        rag_context = get_rag_context(user_message)
        
        # Enhance system instruction with RAG context if available
        if rag_context and rag_context not in ["General conversation - no medical context needed.", 
                                             "RAG system not available.", 
                                             "General medical question - using standard knowledge."]:
            enhanced_system_instruction = system_instruction + f"\n\nRelevant medical context:\n{rag_context}"
        else:
            enhanced_system_instruction = system_instruction
        
        # Build the prompt with clinical data if provided
        prompt = user_message
        content_parts = [{"type": "text", "text": prompt}]
        
        # Add images if provided (max 6 images)
        if image_paths:
            for image_path in image_paths[:6]:
                image_b64 = encode_image_to_base64(image_path)
                if image_b64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    })
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": enhanced_system_instruction},
            {"role": "user", "content": content_parts}
        ]
        
        # Call LLM
        completion = client.chat.completions.create(
            model="unsloth/medgemma-4b-it-GGUF",
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating radiology report: {e}")
        return f"Sorry, I encountered an error while generating the report: {str(e)}"
    finally:
        # Clean up temporary files
        if temp_files_to_cleanup:
            cleanup_temp_files(temp_files_to_cleanup)
   
# ====== Chat Memory Helpers ======
MAX_TURNS = 12  # เก็บล่าสุด 12 เทิร์น (24 ข้อความ: user+model)

def _get_turns(session):
    """
    ดึง history ที่เราเก็บใน session: [{"role":"user|model","content":"..."}]
    """
    return session.get("chat_history", [])

def _save_turns(session, turns):
    """
    เซฟเฉพาะเทิร์นล่าสุดไม่เกิน MAX_TURNS
    """
    trimmed = turns[-MAX_TURNS*2:]  # user+model = *2
    session["chat_history"] = trimmed
    session.modified = True

def _to_openai_history(turns):
    converted_turns = []
    for t in turns:
        role = "assistant" if t["role"] == "model" else t["role"]
        converted_turns.append({"role": role, "content": t["content"]})
    return converted_turns

def _append_turn(session, role, content):
    turns = _get_turns(session)
    role = "assistant" if role == "model" else role
    turns.append({"role": role, "content": content})
    _save_turns(session, turns)

def _reset_chat(session):
    session["chat_history"] = []
    session.modified = True
    
# ====== Core LLM Call with Memory ======
def call_medgemma_rag(session, user_prompt: str, uploaded_files=None, clinical_data=None) -> str:
    """Medical RAG chat with image analysis capability"""
    temp_image_paths = []
    
    try:
        # Handle uploaded images
        if uploaded_files:
            temp_image_paths = handle_uploaded_images(uploaded_files)
        
        # Generate response using medical RAG
        response = generate_radiology_report(user_prompt, temp_image_paths, clinical_data)
        
        # Store in chat history
        _append_turn(session, "user", user_prompt)
        _append_turn(session, "assistant", response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in medical RAG chat: {e}")
        return f"Sorry, I encountered an error in medical analysis: {str(e)}"
    finally:
        # Clean up temporary files
        if temp_image_paths:
            cleanup_temp_files(temp_image_paths)

# ====== Django View ======
@require_http_methods(["GET", "POST"])
@login_required
def chatbot(request):
    #ทำให้เวลาเราดูข้อมูลที่เราเคย Chat ตงกับ User
    chats = Chat.objects.filter(user=request.user)

    # reset history ถ้าผู้ใช้ส่งคำสั่งพิเศษ
    if request.method == "POST" and request.POST.get("reset") == "1":
        _reset_chat(request.session)
        return JsonResponse({"ok": True, "message": "Chat history has been reset."})

    if request.method == 'POST':
        message = request.POST.get('message')
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        # ตัวอย่าง: รองรับพิมพ์ "/reset"
        if message.strip().lower() == "/reset":
            _reset_chat(request.session)
            return JsonResponse({"ok": True, "message": "Chat history has been reset."})

        # Handle file uploads
        uploaded_files = request.FILES.getlist('medical_images') if 'medical_images' in request.FILES else []

        # RAG medical queries
        temp_image_paths = []
        try:
            # Save each uploaded file to temporary location
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
                    for chunk in uploaded_file.chunks():
                        temp_file.write(chunk)
                    temp_image_paths.append(temp_file.name)
            
            # Generate response using medical RAG
            response = generate_radiology_report(message, temp_image_paths)
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            response = f"Error processing your request: {str(e)}"
        
        finally:
            # Clean up temporary files
            for temp_path in temp_image_paths:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {temp_path}: {e}")

        # Save to database
        chat = Chat(
            user=request.user, 
            message=message, 
            response=response, 
            created_at=timezone.now(),
        )

        chat.save()

        return JsonResponse({
            'message': message, 
            'response': response,
            'has_images': len(uploaded_files) > 0,
            'image_count': len(uploaded_files)
        })

    # GET -> render UI
    return render(request, 'chatbot.html')

# ===== Authentication Views =====
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username  = request.POST.get('username','').strip()
        email     = request.POST.get('email','').strip()
        password1 = request.POST.get('password1','')
        password2 = request.POST.get('password2','')

        if password1 != password2:
            return render(request, 'register.html', {'error_message': 'Passwords do not match'})

        User = get_user_model()
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error_message': 'Username is already taken'})

        try:
            validate_password(password1)   # ถ้ามีตัวตรวจความแข็งแรงของรหัสผ่าน จะได้ข้อความที่ชัดเจน
        except ValidationError as e:
            return render(request, 'register.html', {'error_message': ' '.join(e.messages)})

        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
        except IntegrityError as e:
            logger.exception("Integrity error")
            return render(request, 'register.html', {'error_message': f'Integrity error: {e}'})
        except Exception as e:
            logger.exception("Create user failed")
            return render(request, 'register.html', {'error_message': f'Create user failed: {e}'})

        try:
            auth.login(request, user)
        except Exception as e:
            logger.exception("Login failed after creating user")
            return render(request, 'register.html', {'error_message': f'Account created but login failed: {e}'})

        return redirect('chatbot')  # ชื่อเส้นทางของคุณถูกแล้ว
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('chatbot')

def folder(request):
    return render(request, 'folder.html')

def histories(request):
    qs = Chat.objects.select_related('user').order_by('-created_at')
    # ถ้าไม่ใช่ superuser ให้เห็นเฉพาะของตัวเอง
    if not request.user.is_superuser:
        qs = qs.filter(user=request.user)
    return render(request, 'histories.html', {'chats': qs})

