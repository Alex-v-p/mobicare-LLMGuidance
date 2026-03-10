# mobicare-LLMGuidance


wsl --install

apt install -y python3-dev python3-distutils    

mkdir -p ~/repos
cp -r /mnt/c/Users/alexa/Pictures/Internship/repos/mobicare-LLMGuidance ~/repos/
cd ~/repos/mobicare-LLMGuidance

wsl pants

{
  "options": {
    "cleaning_strategy": "deep",
    "cleaning_params": {},
    "chunking_strategy": "naive",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    }
  }
}


{
  "options": {
    "cleaning_strategy": "deep",
    "chunking_strategy": "page_indexed",
    "chunking_params": {}
  }
}


{
  "options": {
    "cleaning_strategy": "deep",
    "chunking_strategy": "late",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    }
  }
}

{
  "options": {
    "cleaning_strategy": "medical_guideline_deep",
    "cleaning_params": {},
    "chunking_strategy": "naive",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    }
  }
}


{
    "request_id": "case-2026-0001",
    "question": "According to the ESC heart failure guidelines, what is the recommended management for a patient with worsening chronic HFrEF who remains symptomatic despite treatment with ACE inhibitor, beta blocker, and MRA?",
    "patient": {
        "values": {
        "age": 67,
        "sex": "male",
        "diagnosis": "chronic heart failure with reduced ejection fraction",
        "ejection_fraction": 30,
        "nyha_class": "III",
        "current_medication": [
            "enalapril",
            "bisoprolol",
            "spironolactone"
            ],
        "blood_pressure_systolic": 110,
        "heart_rate": 78,
        "egfr": 58,
        "recent_hospitalization": true
        }
    },
    "options": {
        "use_retrieval": true,
        "top_k": 5,
        "temperature": 0.2,
        "max_tokens": 300,
        "use_example_response": false,
        "callback_url": "https://example.com/clinical-callback",
        "callback_headers": {
        "Authorization": "Bearer example-token",
        "Content-Type": "application/json"
        }
    }
}



Health URLs

API: http://localhost:8000/health

Inference: http://localhost:8001/health

Qdrant: http://localhost:6333

MinIO console: http://localhost:9001

Ollama: http://localhost:11434