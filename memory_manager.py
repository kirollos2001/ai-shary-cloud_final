import os
import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field
import variables

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = variables.GEMINI_API_KEY

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model=variables.GEMINI_MODEL_NAME,
    google_api_key=variables.GEMINI_API_KEY,
    temperature=0.7
)

class ConversationSummaryMemory:
    """Custom conversation summary memory using LangChain 0.3.x API"""
    
    def __init__(self, llm, max_token_limit=500):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.buffer = []
        
        self.summary_prompt = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="Summarize the following conversation in 50 words or less:\n\n{summary}\n\n{new_lines}"
        )
        
        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()
    
    @property
    def memory_variables(self):
        return ["history"]
    
    def load_memory_variables(self, inputs):
        if not self.buffer:
            return {"history": ""}
        
        # Create summary from buffer
        conversation_text = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in self.buffer
        ])
        
        try:
            summary = self.summary_chain.invoke({
                "summary": "",
                "new_lines": conversation_text
            })
            return {"history": summary}
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {"history": conversation_text}
    
    def save_context(self, inputs, outputs):
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", outputs.get("response", ""))
        
        self.buffer.append(HumanMessage(content=user_input))
        self.buffer.append(AIMessage(content=ai_output))
        
        # Keep buffer within token limit (rough estimation)
        if len(self.buffer) > 20:  # Simple limit
            self.buffer = self.buffer[-20:]
    
    def clear(self):
        self.buffer.clear()

class HybridEntityMemory:
    """Hybrid entity memory: rule-based for common entities, LLM for complex ones"""
    
    def __init__(self, llm, max_token_limit=500, k=5):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.k = k
        self.entities = {}
        self.buffer = []
        self.llm_call_count = 0  # Track API calls
        
        # Rule-based patterns for common entities (no API calls)
        self.rule_patterns = {
            'name': [
                r'أنا\s+([^\s]+(?:\s+[^\s]+)*)',  # Arabic: "أنا أحمد محمد"
                r'اسمي\s+([^\s]+(?:\s+[^\s]+)*)',  # Arabic: "اسمي أحمد محمد"
                r'my name is\s+([^\s]+(?:\s+[^\s]+)*)',  # English: "my name is John Smith"
                r'i am\s+([^\s]+(?:\s+[^\s]+)*)',  # English: "I am John Smith"
            ],
            'phone': [
                r'(\d{11})',  # Egyptian phone: 01234567890
                r'(\d{3}-\d{3}-\d{4})',  # US phone: 123-456-7890
                r'رقم\s+تليفوني\s+(\d+)',  # Arabic: "رقم تليفوني 01234567890"
            ],
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'اميلي\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # Arabic
            ],
            'budget': [
                r'(\d+)\s*مليون\s*جنيه',  # Arabic: "5 مليون جنيه"
                r'\$(\d+(?:,\d{3})*)',  # English: "$500,000"
                r'ميزانيتي\s+(\d+)',  # Arabic: "ميزانيتي 5"
            ],
            'location': [
                r'في\s+([^\s]+(?:\s+[^\s]+)*)',  # Arabic: "في التجمع الخامس"
                r'in\s+([^\s]+(?:\s+[^\s]+)*)',  # English: "in Dubai Marina"
            ],
            'bedrooms': [
                r'(\d+)\s*غرف\s*نوم',  # Arabic: "3 غرف نوم"
                r'(\d+)-bedroom',  # English: "2-bedroom"
            ],
            'area': [
                r'(\d+)\s*متر\s*مربع',  # Arabic: "150 متر مربع"
                r'(\d+)\s*sqm',  # English: "150 sqm"
            ]
        }
        
        # LLM prompt for complex entities (only when needed)
        self.complex_entity_prompt = PromptTemplate(
            input_variables=["conversation", "existing_entities"],
            template="""Extract complex entities and preferences from this conversation that are NOT already captured in the existing entities.

Existing entities: {existing_entities}

Conversation:
{conversation}

Extract ONLY complex entities like:
- Property preferences (style, features, amenities)
- Timeline preferences (when they want to move, construction status)
- Special requirements (accessibility, parking, etc.)
- Investment goals
- Family size/composition
- Work location/commute preferences

Return only valid JSON with new entities:"""
        )
        
        self.complex_entity_chain = self.complex_entity_prompt | self.llm | StrOutputParser()
    
    @property
    def memory_variables(self):
        return ["entities"]
    
    def load_memory_variables(self, inputs):
        return {"entities": self.entities}
    
    def save_context(self, inputs, outputs):
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", outputs.get("response", ""))
        
        self.buffer.append(HumanMessage(content=user_input))
        self.buffer.append(AIMessage(content=ai_output))
        
        # Step 1: Extract common entities using rules (no API call)
        self._extract_rule_based_entities(user_input)
        self._extract_rule_based_entities(ai_output)
        
        # Step 2: Extract complex entities using LLM (only if we have enough context)
        if len(self.buffer) >= 4:  # At least 2 exchanges
            self._extract_complex_entities()
    
    def _extract_rule_based_entities(self, text):
        """Extract common entities using regex patterns (no API calls)"""
        for entity_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if entity_type not in self.entities:
                        self.entities[entity_type] = []
                    self.entities[entity_type].extend(matches)
                    # Remove duplicates
                    self.entities[entity_type] = list(set(self.entities[entity_type]))
    
    def _extract_complex_entities(self):
        """Extract complex entities using LLM (API call)"""
        # Only call LLM if we don't have complex entities yet or every 3 exchanges
        if self.llm_call_count > 0 and self.llm_call_count % 3 != 0:
            return
            
        conversation_text = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in self.buffer[-4:]  # Last 2 exchanges
        ])
        
        try:
            entity_response = self.complex_entity_chain.invoke({
                "conversation": conversation_text,
                "existing_entities": json.dumps(self.entities, indent=2)
            })
            
            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', entity_response, re.DOTALL)
            if json_match:
                new_entities = json.loads(json_match.group())
                self.entities.update(new_entities)
                
                # Keep only top k entities
                if len(self.entities) > self.k:
                    # Keep the most recent ones
                    keys_to_keep = list(self.entities.keys())[-self.k:]
                    self.entities = {k: self.entities[k] for k in keys_to_keep}
                
                self.llm_call_count += 1
                print(f"🔍 LLM entity extraction call #{self.llm_call_count}")
                
        except Exception as e:
            print(f"Error extracting complex entities: {e}")
    
    def clear(self):
        self.entities.clear()
        self.buffer.clear()
        self.llm_call_count = 0

class CombinedConversationMemory:
    """Combined memory that uses both summary and entity memory"""
    
    def __init__(self, summary_memory, entity_memory):
        self.summary_memory = summary_memory
        self.entity_memory = entity_memory

    @property
    def memory_variables(self):
        return ["history", "entities_str"]

    def load_memory_variables(self, inputs):
        # Ensure inputs is not empty
        if not inputs or not isinstance(inputs, dict):
            inputs = {"input": ""}
        
        # Ensure "input" key exists
        if "input" not in inputs:
            inputs["input"] = ""
            
        summary = self.summary_memory.load_memory_variables(inputs).get("history", "")
        ent = self.entity_memory.load_memory_variables(inputs).get("entities", {})
        return {
            "history": summary,
            "entities_str": json.dumps(ent, indent=2)
        }

    def save_context(self, inputs, outputs):
        # Ensure inputs is not empty
        if not inputs or not isinstance(inputs, dict):
            inputs = {"input": ""}
        
        # Ensure "input" key exists
        if "input" not in inputs:
            inputs["input"] = ""
            
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", outputs.get("response", ""))
        
        # Save summary
        self.summary_memory.save_context({"input": user_input}, {"output": ai_output})
        # Save entities (hybrid approach)
        self.entity_memory.save_context({"input": user_input}, {"output": ai_output})

    def add_message_to_memory(self, message, sender="user"):
        """Add a message to memory (for backward compatibility)"""
        if sender == "user":
            self.save_context({"input": message}, {"output": ""})
        else:
            # For AI messages, we need a user input to pair with
            self.save_context({"input": "AI response"}, {"output": message})

    def update_user_preferences(self, preferences):
        """Update user preferences in entity memory (for backward compatibility)"""
        # This method is for backward compatibility
        # The actual preference extraction happens in the save_context method
        pass

    def update_search_results(self, session_id, results):
        """Update search results (for backward compatibility)"""
        # This method is for backward compatibility
        # Search results are not stored in this memory manager
        pass

    def clear(self):
        # Clear both memories
        self.summary_memory.clear()
        self.entity_memory.clear()

# Initialize memories
summary_memory = ConversationSummaryMemory(llm)
entity_memory = HybridEntityMemory(llm)

# Build memory manager
memory_manager = CombinedConversationMemory(summary_memory, entity_memory)
