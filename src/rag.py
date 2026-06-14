from qdrant_client.models import VectorParams, PointStruct

system_prompt = """You are a compassionate mental health support assistant.
        Use the following retrieved counseling examples as guidance for tone and content,
        but DO NOT copy them verbatim — synthesize your own empathetic, grounded response.
        If the user mentions self-harm or crisis, gently encourage them to reach out to a crisis helpline
        and to a trusted person, in addition to your response.
        {emotion_note}

        {context_block}
        """

class RAG:
    def __init__(self, db_client, LLM_client, embedder):
        self.db_client = db_client
        self.LLM_client = LLM_client
        self.embedder = embedder
        self.system_prompt = system_prompt


    def create_collection(self, collection_name:str, size:int, distance):
        collections = self.db_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            self.db_client.delete_collection(collection_name)

        self.db_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance)
        )


    def ingest(self, dataset, collection_name:str, batch_size:int):
        points = []

        for i, row in enumerate(dataset["train"]):
            vector = self.embedder.encode(row["Context"]).tolist()
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={"context": row["Context"], "response": row["Response"]}
            ))

        for i in range(0, len(points), batch_size):
            self.db_client.upsert(collection_name=collection_name, points=points[i:i+batch_size])


    def retrieve(self, query, collection_name, top_k=3):
        query_vector = self.embedder.encode(query).tolist()
        results = self.db_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        return [
            {"context": hit.payload["context"], "response": hit.payload["response"], "score": hit.score} 
            for hit in results.points
        ]


    def generate(self, user_query:str, model_name:str, emotion=None, collection_name:str="mental_health_kb"):
        relevant_docs = self.retrieve(user_query, collection_name)
        
        context_block = "\n\n".join(
            f"Similar past conversation:\nQ: {doc['context']}\nA: {doc['response']}" for doc in relevant_docs
        )
        
        emotion_note = f"\nThe user seems to be feeling: {emotion}. Adjust your tone accordingly." if emotion else ""

        response = self.LLM_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.system_prompt.format(
                        emotion_note=emotion_note, context_block=context_block
                    )
                },
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content