import google.generativeai as genai
import os
import logging
import json
import time
import config
import functions
import variables
from memory_manager import memory_manager


def check_gemini_setup():
    """Check if Gemini API is properly configured"""
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        logging.info("✅ Gemini API is properly configured.")
        return True
    except Exception as e:
        logging.error(f"❌ Error configuring Gemini API: {e}")
        return False

import asyncio

# Legacy conversation memory (for backward compatibility)
conversation_memory = {}

def get_conversation_context(session_id):
    """Get conversation context for a session using LangChain memory"""
    try:
        mem = memory_manager(session_id)
        memory_vars = mem.load_memory_variables({})
        
        # Get legacy context for backward compatibility
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                "user_preferences": {},
                "previous_messages": [],
                "current_search_results": [],
                "user_info": {}
            }
        
        # Combine LangChain memory with legacy context
        context = conversation_memory[session_id].copy()
        context.update({
            "summary": memory_vars.get("history", ""),
            "entities": memory_vars.get("entities", {})
        })
        
        return context
    except Exception as e:
        logging.error(f"Error getting conversation context: {e}")
        # Return default context on error
        return {
            "user_preferences": {},
            "previous_messages": [],
            "current_search_results": [],
            "user_info": {},
            "summary": "",
            "entities": {}
        }

def update_conversation_context(session_id, key, value):
    """Update conversation context using LangChain memory"""
    try:
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                "user_preferences": {},
                "previous_messages": [],
                "current_search_results": [],
                "user_info": {},
            }
        conversation_memory[session_id][key] = value
    except Exception as e:
        logging.error(f"Error updating conversation context: {e}")
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                "user_preferences": {},
                "previous_messages": [],
                "current_search_results": [],
                "user_info": {},
            }
        conversation_memory[session_id][key] = value

def extract_user_preferences_from_message(user_message):
    """Extract user preferences from message and update context"""
    preferences = functions.extract_client_preferences(user_message)
    
    # Additional extraction for specific keywords
    user_message_lower = user_message.lower()
    
    # Extract budget if mentioned in millions
    if "مليون" in user_message_lower:
        import re
        budget_match = re.search(r"(\d+)\s*مليون", user_message_lower)
        if budget_match:
            preferences["budget"] = int(budget_match.group(1)) * 1000000
    
    # Extract bedrooms if mentioned
    if "اوض" in user_message_lower or "غرف" in user_message_lower:
        import re
        bedrooms_match = re.search(r"(\d+)\s*(?:اوض|غرف)", user_message_lower)
        if bedrooms_match:
            preferences["bedrooms"] = int(bedrooms_match.group(1))
    
    # Extract area if mentioned in meters
    if "متر" in user_message_lower:
        import re
        area_match = re.search(r"(\d+)\s*متر", user_message_lower)
        if area_match:
            preferences["area"] = int(area_match.group(1))
    
    # Extract location from context or message
    location_keywords = ["التجمع الخامس", "التجمع السادس", "6 اكتوبر", "الشيخ زايد", "ماونتن فيو", "العاصمة"]
    for location in location_keywords:
        if location in user_message_lower:
            preferences["location"] = location
            break
    
    # Extract delivery type
    if "فوري" in user_message_lower or "استلام فوري" in user_message_lower:
        preferences["delivery_type"] = "فوري"
    elif "تحت الإنشاء" in user_message_lower:
        preferences["delivery_type"] = "تحت الإنشاء"
    
    # Extract compound preference
    if "كومبوند" in user_message_lower or "كمبوند" in user_message_lower:
        preferences["compound"] = True
    
    return preferences

async def process_gemini_response_async(model, user_message, session_id=None):
    """Process Gemini response and handle automatic function calls with LangChain memory"""
    try:
        mem = memory_manager(session_id)
        # Save user message to LangChain memory with proper error handling
        try:
            mem.save_context({"input": user_message}, {"output": ""})
        except Exception as e:
            logging.warning(f"Failed to save to memory manager: {e}")
        
        # Get conversation context using LangChain memory
        context = get_conversation_context(session_id)

        # Extract new preferences from current message
        new_preferences = extract_user_preferences_from_message(user_message)

        if new_preferences:
            merged_preferences = {
                **context.get("preferences", {}),
                **{k: v for k, v in new_preferences.items() if v}
            }
            update_conversation_context(session_id, "preferences", merged_preferences)

        # Get updated context after preference update
        context = get_conversation_context(session_id)
        
        # Create enhanced context-aware prompt with LangChain memory
        context_prompt = ""
        
        # Add conversation summary
        if context.get("summary"):
            context_prompt += f"\n### ملخص المحادثة السابقة:\n{context['summary']}\n"
        
        # Add entity memory (extracted preferences and entities)
        if context.get("entities"):
            context_prompt += f"\n### المعلومات المحفوظة عن المستخدم:\n"
            for key, value in context["entities"].items():
                context_prompt += f"- {key}: {value}\n"
        
        # Add current preferences
        if context.get("preferences", {}):
            context_prompt += f"\n### تفضيلات المستخدم الحالية:\n"
            for key, value in context["preferences"].items():
                context_prompt += f"- {key}: {value}\n"
        
        # Create the full prompt with system instructions and enhanced context
        system_prompt = f"{config.assistant_instructions}\n\n### Examples:\n{config.examples}"
        full_prompt = f"{system_prompt}\n\n{context_prompt}\n\n### رسالة المستخدم:\n{user_message}"
        
        # Generate response from Gemini with automatic function calling
        logging.info(f"Generating response for user message: {user_message[:100]}...")
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        
        # Check if response contains function calls
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Handle automatic function call
                        function_name = part.function_call.name
                        function_args = part.function_call.args
                        
                        logging.info(f"🔧 Automatic function call detected: {function_name}")
                        logging.info(f"🔧 Function arguments: {function_args}")
                        
                        # Execute search functions immediately (no wait message needed)
                        if function_name in ["property_search", "search_new_launches", "insight_search"]:
                            # Continue to execute the function immediately
                            pass
                        
                        try:
                            # Execute the function
                            function_to_call = getattr(functions, function_name)

                            # Add session_id to arguments if needed
                            if function_name == "schedule_viewing" and session_id:
                                function_args["conversation_id"] = session_id
                            
                            # Add session_id for get_more_units
                            if function_name == "get_more_units" and session_id:
                                function_args["session_id"] = session_id

                            # Add client info for scheduling
                            if function_name == "schedule_viewing":
                                client_info = config.client_sessions.get(session_id, {})
                                function_args.update({
                                    "client_id": client_info.get("user_id", 1),
                                    "name": client_info.get("name", "Unknown"),
                                    "phone": client_info.get("phone", "Not Provided"),
                                    "email": client_info.get("email", "Not Provided")
                                })

                            # Augment property_search with compound from conversation preferences when missing
                            if function_name == "property_search":
                                # Add session_id for progressive search
                                if session_id and not function_args.get("session_id"):
                                    function_args["session_id"] = session_id
                                
                                try:
                                    client_info = config.client_sessions.get(session_id, {})
                                    user_id = client_info.get("user_id")
                                    if user_id:
                                        prefs = functions.get_conversation_preferences(session_id, user_id)
                                        compound_pref = (
                                            prefs.get("compound_name")
                                            or prefs.get("compound")
                                        )
                                        if compound_pref and not function_args.get("compound") and not function_args.get("compound_name"):
                                            function_args["compound"] = compound_pref
                                except Exception as _e:
                                    logging.warning(f"Could not augment property_search with compound: {_e}")

                            # For insight queries: ensure we pass inferred location/type if present in context
                            if function_name == "insight_search":
                                try:
                                    client_info = config.client_sessions.get(session_id, {})
                                    user_id = client_info.get("user_id")
                                    if user_id:
                                        prefs = functions.get_conversation_preferences(session_id, user_id)
                                        if prefs.get("location") and not function_args.get("location"):
                                            function_args["location"] = prefs.get("location")
                                        if prefs.get("property_type") and not function_args.get("property_type"):
                                            function_args["property_type"] = prefs.get("property_type")
                                except Exception as _e:
                                    logging.warning(f"Could not augment insight_search with context: {_e}")

                            output = function_to_call(function_args)
                            logging.info(f"✅ Function {function_name} executed successfully!")
                            
                            # Store search results in context if it's a property search
                            if function_name == "property_search" and isinstance(output, dict) and "results" in output:
                                logging.info(f"🔍 Property search results: {len(output['results'])} properties found")
                            
                            return {
                                "function_output": output,
                                "function_name": function_name
                            }
                            
                        except Exception as e:
                            logging.error(f"🚫 Error executing function {function_name}: {e}")
                            return {
                                "error": f"Error executing function {function_name}: {str(e)}"
                            }
        
        # If no function call, get the text response
        response_text = response.text if response.text else "❌ لم يتم استلام رد من المساعد."
        logging.info(f"Gemini text response: {response_text[:200]}...")
        
        # Save AI response to LangChain memory with proper error handling
        try:
            mem.save_context({"input": user_message}, {"output": response_text})
        except Exception as e:
            logging.warning(f"Failed to save AI response to memory manager: {e}")
        
        # Return the text response
        return {
            "text_response": response_text
        }
        
    except Exception as e:
        logging.error(f"🚫 Error processing Gemini response: {e}")
        logging.error(f"🚫 Error type: {type(e)}")
        logging.error(f"🚫 Error details: {str(e)}")
        return {
            "error": f"Error processing response: {str(e)}"
        }

def get_resource_files():
    """Get list of resource files for Gemini (if needed)"""
    file_paths = []
    resources_folder = 'resources'
    if os.path.exists(resources_folder):
        for filename in os.listdir(resources_folder):
            file_path = os.path.join(resources_folder, filename)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
    return file_paths
