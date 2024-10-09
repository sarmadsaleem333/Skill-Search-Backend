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
                # if distances[0][j] > 1.0:
                #       results.append({
                #     "skill_id": skill_id,
                #     "skill_name": matched_skill_name,
                #     "distance": distances[0][j]
                #         })

            end_time = time.time()
            print("Time taken for search:", end_time - start_time)
            
            return Response(results, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AddSkillView(APIView):
    """API to add a new skill to the FAISS index and JSON file."""
    
    def post(self, request):
        skill_name = request.data.get('skill_name')
     
        skill_id = request.data.get('skill_id')


        if not skill_name or not skill_id:
            return Response({"error": "Both skill_name and skill_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:

            INDEX_FILE = os.path.join(os.path.dirname(__file__), "faiss_skills_index")
                # Load data from JSON file located in the same directory as this script
            skills_file_path = os.path.join(os.path.dirname(__file__), 'skills.json')
            with open(skills_file_path, 'r') as file:
                data = json.load(file)
        
                
            skill_ids = [skill['skill_id'] for skill in data]

            if os.path.exists(INDEX_FILE):
                index = faiss.read_index(INDEX_FILE)
            else:
                return Response({"error": f"No FAISS index found at {INDEX_FILE}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Generate embedding for the new skill
            embedding_vector = embeddings.embed_query(skill_name)
            embedding_vector = np.array(embedding_vector).astype('float32')

            # Add new skill to index
            index.add(np.expand_dims(embedding_vector, axis=0))

            # Create a new skill entry
            new_skill = {"skill_name": skill_name, "skill_id": skill_id}

            # Append the new skill to the existing skills data
            data.append(new_skill)

            # Save updated skills data back to the JSON file
            with open(skills_file_path, 'w') as file:
                json.dump(data, file, indent=4)  # Save with indentation for readability

            # Save the updated FAISS index back to the file
            faiss.write_index(index, INDEX_FILE)

            return Response({"message": f"Skill '{skill_name}' added with skill_id {skill_id}."}, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)





class DeleteSkillView(APIView):
    """API to delete a skill from the FAISS index and JSON file."""
    
    def delete(self, request, skill_id):
        try:
           
            skills_file_path = os.path.join(os.path.dirname(__file__), 'skills.json')      
            with open(skills_file_path, 'r') as file:
                data = json.load(file)

            
            # Find the skill in the JSON file
            skill_to_delete = None
            for skill in data:
                print(skill['skill_id'],skill_id)
                if str(skill['skill_id']) == skill_id:
                    skill_to_delete = skill
                    break
            if skill_to_delete is None:
                return Response({"message": f"Skill does not exist."}, status=status.HTTP_404_NOT_FOUND)

            # Get the index of the skill to delete
            pos = data.index(skill_to_delete)

            # Remove the skill from the list
            data.pop(pos)

            # Save updated skills data back to the JSON file
            with open(skills_file_path, 'w') as file:
                json.dump(data, file, indent=4)  # Save with indentation for readability

            dim = 768  # Dimension of the embeddings
            index_new = faiss.IndexFlatL2(dim)

            # Rebuild the index without the deleted skill
            for skill in data:
                skill_name = skill['skill_name']
                # Generate embedding for the existing skill
                embedding_vector = embeddings.embed_query(skill_name)
                embedding_vector = np.array(embedding_vector).astype('float32')
                index_new.add(np.expand_dims(embedding_vector, axis=0))

            # Save the updated FAISS index back to the file
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "faiss_skills_index")
            faiss.write_index(index_new, INDEX_FILE)

            return Response({"message": f"Skill with skill_id {skill_id} deleted."}, status=status.HTTP_204_NO_CONTENT)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

