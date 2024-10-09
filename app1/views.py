from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import faiss
import json
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
import time

# Initialize HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

class SkillSearchView(APIView):
    """API to search skills using FAISS and return the closest matching skills."""
    
    def get(self, request):
        try:
            # Get the skill name from query parameters
            skill_name = request.query_params.get('skill_name', None)
            
            if not skill_name:
                return Response({"error": "skill_name parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Start skill search logic
            start_time = time.time()

            # Load data from JSON file located in the same directory as this script
            skills_file_path = os.path.join(os.path.dirname(__file__), 'skills.json')
            with open(skills_file_path, 'r') as file:
                data = json.load(file)

            applied_skills = data
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "faiss_skills_index")
            skill_ids = [skill['skill_id'] for skill in data]

            if os.path.exists(INDEX_FILE):
                index = faiss.read_index(INDEX_FILE)
            else:
                return Response({"error": f"No FAISS index found at {INDEX_FILE}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Get the query embedding
            query_embedding = embeddings.embed_query(skill_name)
            query_embedding_np = np.array(query_embedding).astype('float32')

            # Search the FAISS index for the closest matches
            distances, indices = index.search(np.expand_dims(query_embedding_np, axis=0), 4)

            # Fetch the results, mapping indices to skill names and IDs
            results = []
            for j, i in enumerate(indices[0]):
                skill_id = skill_ids[i]
                matched_skill_name = next(skill['skill_name'] for skill in applied_skills if skill['skill_id'] == skill_id)

                results.append({
                    "skill_id": skill_id,
                    "skill_name": matched_skill_name,
                    "distance": distances[0][j]
                })

            end_time = time.time()
            print("Time taken for search:", end_time - start_time)

            return Response(results, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
