import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("FastNewsParser")

class FastNewsParser:
    def __init__(self):
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            logger.warning("LLM_API_KEY not found in environment. Parser will fail.")
        self.client = OpenAI(api_key=api_key)
        
        self.system_prompt = (
            "You are a high-frequency sports data parser. "
            "Extract the NBA player's name and their playing status from the following tweet. "
            "Respond ONLY in valid JSON format with the keys 'player_name' and 'status'. "
            "Standardize 'status' to one of: 'IN', 'OUT', 'QUESTIONABLE', 'PROBABLE', or 'UNKNOWN'."
        )

    def parse_tweet(self, tweet_text: str) -> dict:
        """Parses unstructured text into a deterministic JSON status."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": tweet_text}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0 # Zero temperature for deterministic output
            )
            
            parsed_json = json.loads(response.choices[0].message.content)
            logger.info(f"Successfully parsed tweet for {parsed_json.get('player_name')}")
            return parsed_json
            
        except Exception as e:
            logger.error(f"Failed to parse tweet: {e}")
            return {"player_name": "unknown", "status": "UNKNOWN"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = FastNewsParser()
    
    sample_tweet = "Dallas Mavericks PR: Luka Doncic (right ankle sprain) has been downgraded to OUT for tonight's game against Denver."
    print(f"Raw Tweet: {sample_tweet}")
    
    result = parser.parse_tweet(sample_tweet)
    print(f"Parsed JSON: {json.dumps(result, indent=2)}")
