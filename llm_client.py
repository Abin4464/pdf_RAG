from groq import Groq

class LLMClient:
    """Generate responses using Groq LLM (FREE)"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response based on query and context"""
        
        system_prompt = """You are a helpful assistant. Answer the user's question based ONLY on the provided context. 
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and accurate."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            print(f"\n✓ Generated response ({len(answer)} characters)")
            return answer
        
        except Exception as e:
            print(f"✗ Error generating response: {str(e)}")
            return f"Error: {str(e)}"