from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import auth
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tempfile
import os
import json
import csv
import base64
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .models import Chat
from bonechat.models import Chat

logger = logging.getLogger(__name__)

# ===== Load ENV & Configure =====
load_dotenv()

# Environment setup
LMS_URL = os.getenv("base_url_lms")
LMS_API = os.getenv("LMS_API")
EMBED_MODEL = "google/embeddinggemma-300m"
CSV_PATH = os.getenv("csv_path")
IMGSET_PATH = os.getenv("dataimage_path")
TEMPLATE_PATH = os.getenv("tempath")

# Initialize OpenAI client
client = OpenAI(base_url=f"{LMS_URL.rstrip('/')}/v1", api_key=LMS_API)

# Initialize embedder
embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

# Load FAISS indices (load once at startup)
try:
    index_text = FAISS.load_local("faiss_text", embedder, allow_dangerous_deserialization=True)
    index_tabular = FAISS.load_local("faiss_tabular", embedder, allow_dangerous_deserialization=True)
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

# ===== RAG Utility Functions =====
def load_system_template(template_path):
    """Load system template from JSON file"""
    try:
        with open(template_path, "r", encoding='utf-8') as f:
            json_template = json.load(f)
        
        system_prompt = (
            "You are a Radiology Expert. You will be given X-ray images, analyze them and always answer "
            "in the following structured format and use based knowledge that provided from retrieval within 512 tokens.\n"
            "Do not reference which sources used, just answer based on format given\n\n"
            "Answer format:\n"
            + json.dumps(json_template, indent=2)
        )
        return system_prompt
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        return "You are a Radiology Expert. Analyze the X-ray images and provide a detailed report."

def get_rag_context(query_prompt):
    """Retrieve relevant context from FAISS indices"""
    if not FAISS_LOADED:
        return "RAG system not available."
    
    try:
        # Query both indices
        query_results_csv = index_tabular.similarity_search(query_prompt, k=3)
        query_results_pdf = index_text.similarity_search(query_prompt, k=3)
        
        # Combine results
        combined_results = []
        
        for doc in query_results_csv:
            combined_results.append(f"Clinical Data: {doc.page_content}")
        
        for doc in query_results_pdf:
            combined_results.append(f"Medical Literature: {doc.page_content[:500]}...")
        
        return "\n".join(combined_results) if combined_results else "No relevant context found."
    
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        return "Error retrieving medical context."

def generate_radiology_report(user_message, image_paths=None, clinical_data=None):
    """Generate radiology report using RAG and image analysis"""
    temp_files_to_cleanup = []
    
    try:
        # Load system template
        system_template = load_system_template(TEMPLATE_PATH)
        
        # Build the prompt with clinical data if provided
        prompt = user_message
        if clinical_data:
            clinical_text = f" Clinical features: RH={clinical_data.get('rh', 'N/A')}, RI={clinical_data.get('ri', 'N/A')}, UV={clinical_data.get('uv', 'N/A')}."
            prompt += clinical_text
        
        # Prepare content parts
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
        
        # Get RAG context
        rag_context = get_rag_context(prompt)
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": content_parts},
            {"role": "assistant", "content": rag_context}
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

# ===== Chat Memory Helpers =====
MAX_TURNS = 12

def _get_turns(session):
    return session.get("chat_history", [])

def _save_turns(session, turns):
    trimmed = turns[-MAX_TURNS*2:]
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

# ===== Core LLM Call Functions =====
def call_openai_with_memory(session, user_prompt: str) -> str:
    """Standard chat with memory"""
    try:
        turns = _get_turns(session)
        openai_history = _to_openai_history(turns)
        
        messages = openai_history + [{"role": "user", "content": user_prompt}]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        _append_turn(session, "user", user_prompt)
        _append_turn(session, "assistant", assistant_response)
        
        return assistant_response
        
    except Exception as e:
        logger.error(f"Error in standard chat: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

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

# ===== Django Views =====
@require_http_methods(["GET", "POST"])
@login_required
def chatbot(request):
    """Main chatbot view with medical RAG capability"""
    chats = Chat.objects.filter(user=request.user)
    
    # Reset chat history if requested
    if request.method == "POST" and request.POST.get("reset") == "1":
        _reset_chat(request.session)
        return JsonResponse({"ok": True, "message": "Chat history has been reset."})

    if request.method == 'POST':
        message = request.POST.get('message', '').strip()
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        # Handle reset command
        if message.lower() == "/reset":
            _reset_chat(request.session)
            return JsonResponse({"ok": True, "message": "Chat history has been reset."})

        # Handle file uploads
        uploaded_files = request.FILES.getlist('medical_images') if 'medical_images' in request.FILES else []
        
        # Extract clinical data if provided
        clinical_data = {}
        if request.POST.get('rh'):
            clinical_data['rh'] = request.POST.get('rh')
        if request.POST.get('ri'):
            clinical_data['ri'] = request.POST.get('ri')
        if request.POST.get('uv'):
            clinical_data['uv'] = request.POST.get('uv')
        
        # Determine which function to use
        has_medical_context = (
            "x-ray" in message.lower() or 
            "radiology" in message.lower() or 
            "fracture" in message.lower() or
            "medical" in message.lower() or
            uploaded_files or
            clinical_data
        )
        
        if has_medical_context:
            # Use medical RAG for medical queries
            response = call_medgemma_rag(
                request.session, 
                message, 
                uploaded_files if uploaded_files else None,
                clinical_data if clinical_data else None
            )
            response_type = "medical"
        else:
            # Use standard chat for general queries
            response = call_openai_with_memory(request.session, message)
            response_type = "general"

        # Save to database
        chat = Chat(
            user=request.user, 
            message=message, 
            response=response, 
            created_at=timezone.now(),
            response_type=response_type
        )
        chat.save()
        
        return JsonResponse({
            'message': message, 
            'response': response,
            'response_type': response_type,
            'has_images': len(uploaded_files) > 0,
            'image_count': len(uploaded_files)
        })

    return render(request, 'chatbot.html', {'chats': chats})

@require_http_methods(["POST"])
@login_required
def medical_analysis_api(request):
    """Dedicated API endpoint for medical analysis with images"""
    if request.method == 'POST':
        message = request.POST.get('message', '').strip()
        if not message:
            return JsonResponse({'error': 'Empty message'}, status=400)
        
        # Handle file uploads
        uploaded_files = request.FILES.getlist('images') if 'images' in request.FILES else []
        
        # Extract clinical data
        clinical_data = {}
        clinical_fields = ['rh', 'ri', 'uv', 'age', 'gender', 'symptoms']
        for field in clinical_fields:
            if request.POST.get(field):
                clinical_data[field] = request.POST.get(field)
        
        try:
            response = call_medgemma_rag(
                request.session, 
                message, 
                uploaded_files if uploaded_files else None,
                clinical_data if clinical_data else None
            )
            
            # Save to database
            chat = Chat(
                user=request.user,
                message=f"[MEDICAL] {message}",
                response=response,
                created_at=timezone.now(),
                response_type="medical_api"
            )
            chat.save()
            
            return JsonResponse({
                'success': True,
                'message': message,
                'response': response,
                'images_processed': len(uploaded_files),
                'clinical_data_provided': bool(clinical_data)
            })
            
        except Exception as e:
            logger.error(f"Medical analysis API error: {e}")
            return JsonResponse({'error': str(e)}, status=500)

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
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')
        
        if password1 != password2:
            return render(request, 'register.html', {'error_message': 'Passwords do not match'})
        
        User = get_user_model()
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error_message': 'Username is already taken'})
        
        try:
            validate_password(password1)
        except ValidationError as e:
            return render(request, 'register.html', {'error_message': ' '.join(e.messages)})
        
        try:
            user = User.objects.create_user(username=username, email=email, password=password1)
            auth.login(request, user)
            return redirect('chatbot')
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return render(request, 'register.html', {'error_message': f'Registration failed: {e}'})
    
    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('chatbot')

def folder(request):
    return render(request, 'folder.html')

@login_required
def histories(request):
    qs = Chat.objects.select_related('user').order_by('-created_at')
    if not request.user.is_superuser:
        qs = qs.filter(user=request.user)
    return render(request, 'histories.html', {'chats': qs})