from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import faiss
import json
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os
import time
import google.generativeai as genai

# Initialize HuggingFaceEmbeddings
# importing the model from langchain_huggingface which will generate embedding for us
embeddings = HuggingFaceEmbeddings()
genai.configure(api_key="")

            # Define the model
model = genai.GenerativeModel("gemini-1.5-flash")



# for applied skills that user have entered and now admin has to do operations on it
# for applied skills using applied_skills_faiss_index and applied_skills.json
class AppliedSkillSearchView(APIView):

    # for admin
    # getting all skills in applied database related to searched skill like for reactjs it comes react javascript etc
    def get(self, request):
        try:
            # Get the skill name from query parameters
            skill_name = request.query_params.get('skill_name', None)
            
            if not skill_name:
                return Response({"error": "skill_name parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
      
            start_time = time.time()

            # Load data from JSON file located in the same directory 
            # using this as db
            skills_file_path = os.path.join(os.path.dirname(__file__), 'applied_skills.json')
            with open(skills_file_path, 'r') as file:
                data = json.load(file)

            applied_skills = data
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "applied_faiss_skills_index")
            skill_ids = [skill['skill_id'] for skill in data]

            if os.path.exists(INDEX_FILE):
                index = faiss.read_index(INDEX_FILE)
            else:
                return Response({"error": f"No FAISS index found at {INDEX_FILE}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Get the query embedding
            query_embedding = embeddings.embed_query(skill_name)
            query_embedding_np = np.array(query_embedding).astype('float32')

            # Search the FAISS index for the closest matches
            distances, indices = index.search(np.expand_dims(query_embedding_np, axis=0), 10)

            # Fetch the results, mapping indices to skill names and IDs
            results = []
            for j, i in enumerate(indices[0]):
                skill_id = skill_ids[i]
                matched_skill_name = next(skill['skill_name'] for skill in applied_skills if skill['skill_id'] == skill_id)

                if distances[0][j] < 0.5:
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



    # for user who apply for skill
    # that skill is stored in database and as well as in faiss indexing 
    # all skills embedding not generated  but new skill embedding is generated and appended to faiss index
    def post(self, request):
        skill_name = request.data.get('skill_name')
     
        skill_id = request.data.get('skill_id')


        if not skill_name or not skill_id:
            return Response({"error": "Both skill_name and skill_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:

            INDEX_FILE = os.path.join(os.path.dirname(__file__), "applied_faiss_skills_index")
            skills_file_path = os.path.join(os.path.dirname(__file__), 'applied_skills.json')
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



    # for admin
    # for deleting the skill from database and faiss indexing when user approve some skill
    # in this case when skill is added all skills embedding again generated and added to faiss index(this is just overhead)
    def delete(self, request, skill_id):
        try:
           
            skills_file_path = os.path.join(os.path.dirname(__file__), 'applied_skills.json')      
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
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "applied_faiss_skills_index")
            faiss.write_index(index_new, INDEX_FILE)

            return Response({"message": f"Skill with skill_id {skill_id} deleted."}, status=status.HTTP_204_NO_CONTENT)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)






# class for handling database approved skills that recommend user for the skills on job title
# for database skills using database_faiss_index and database_skills.json
class ApprovedSkillSearchView(APIView):

    # for user
    # api for user when user enter job title and it will return the skills related to that job title which is in database
    def get(self, request):
        try:

            job_title = request.query_params.get('job_title', None)
            print("mss",job_title)
            if not job_title:
                return Response({"error": "job_title parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate content with a specific request for an array of skills
            prompt = f"""
                You are a helpful assistant that generates a comprehensive job description based on a job title provided by the user.
                Your response should include the following sections:
                Your response **must** strictly follow this format to be valid JSON with no additional newlines or special characters:
                1. "We are" statement introducing the job description not specifying our company, only providing a job description summary for the given title for 3-4 lines.
                2. Key responsibilities (5-6 bullet points).
                3. Requirements (5-6 bullet points).
                4. Random 6-7 benefits related to the job like monthly team dinners, outings, etc.
                5. Provide a list of 5 important skills  for this job title in a JSON array format (not in HTML).

                The job title is: {job_title}.
                
                Provide the result in Html Rich Text format with the following fields:
                - Description
                - Responsibilities
                - Requirements
                - Benefits

                Additionally, provide a list of skills in a JSON array format:
                - Skills

                Do not add /n in the response.

                Response should be in the following style:
                "description": "p tag description here"
                "responsibilities": "list tag responsibilities here"
                "requirements": "list tag requirements here"
                "benefits": "list tag benefits here"
                "skills": ["skill1", "skill2", "skill3", ...]
                
            """

            
            # Make the request to the model
            response = model.generate_content(prompt)
            print(response.text,"ok")
            # Convert the JSON string into a Python dictionary (JSON object)
            json_object = json.loads(response.text)
            print(json_object["skills"])

        

            
            
            skill_names=json_object["skills"]

            start_time = time.time()

            # Load data from JSON file located in the same directory as this script
            skills_file_path = os.path.join(os.path.dirname(__file__), 'database_skills.json')
            with open(skills_file_path, 'r') as file:
                data = json.load(file)

            applied_skills = data
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "database_faiss_skills_index")
            skill_ids = [skill['skill_id'] for skill in data]

            if os.path.exists(INDEX_FILE):
                index = faiss.read_index(INDEX_FILE)
            else:
                return Response({"error": f"No FAISS index found at {INDEX_FILE}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


            # Get the query embedding

            results = []
            seen_skills = set()  


             # Iterate through each skill in the skill_names array
            for skill_name in skill_names:
                print(skill_name)

                # Get the query embedding
                query_embedding = embeddings.embed_query(skill_name)
                query_embedding_np = np.array(query_embedding).astype('float32')
            
                # Search the FAISS index
 
            
                distances, indices = index.search(np.expand_dims(query_embedding_np, axis=0), 10)

                for j, i in enumerate(indices[0]):
                    skill_id = skill_ids[i]
                    skill_name_result = next(skill['skill_name'] for skill in applied_skills if skill['skill_id'] == skill_id)

                    if skill_name_result not in seen_skills: 
                        results.append({
                            "skill_id": skill_id,
                            "skill_name": skill_name_result,
                            "distance": distances[0][j]
                        })
                        seen_skills.add(skill_name_result)
            end_time = time.time()
            print("Time taken for search:", end_time - start_time)
            results = sorted(results, key=lambda x: x['distance'])




            # if we wanna get top k skills
            # topk=20

            # if len(results) > 20:
            #     results = results[:topk]

            return Response(results, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)




# for admin
# when admin wants to add new skill to database
# that skill is added in db as well as in faiss indexing
# all skills embedding not generated  but new skill embedding is generated and appended to faiss index


    def post(self, request):

        skill_name = request.data.get('skill_name')
     
        skill_id = request.data.get('skill_id')



        if not skill_name or not skill_id:
            return Response({"error": "Both skill_name and skill_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:

            INDEX_FILE = os.path.join(os.path.dirname(__file__), "database_faiss_skills_index")
                # Load data from JSON file located in the same directory as this script
            skills_file_path = os.path.join(os.path.dirname(__file__), 'database_skills.json')
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



# for admin
# for deleteing skills from database and faiss indexing
# in this case when skill is added all skills embedding again generated and added to faiss index(this is just overhead)
    def delete(self, request, skill_id):
        try:
           
            skills_file_path = os.path.join(os.path.dirname(__file__), 'database_skills.json')      
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
            INDEX_FILE = os.path.join(os.path.dirname(__file__), "database_faiss_skills_index")
            faiss.write_index(index_new, INDEX_FILE)

            return Response({"message": f"Skill with skill_id {skill_id} deleted."}, status=status.HTTP_204_NO_CONTENT)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    