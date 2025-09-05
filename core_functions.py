import google.generativeai as genai
import os
import logging
import json
import time
import config
import functions
import variables
from memory_manager import memory_manager
from session_store import get_session
import re

def parse_function_call_from_text(response_text):
    """
    Parse function calls from LLM text response and execute them
    Returns: dict with function_output and function_name if found, None otherwise
    """
    try:
        # Check for unit detail requests first (special case)
        unit_detail_patterns = [
            r'تفاصيل الوحدة.*?(\d+)',
            r'تفاصيل.*?(\d+)',
            r'وحدة.*?(\d+)',
            r'unit.*?(\d+)',
            r'id.*?(\d+)',
            r'رقم.*?(\d+)'
        ]
        
        for pattern in unit_detail_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                unit_id = match.group(1)
                logging.info(f"🔧 Detected unit detail request for ID: {unit_id}")
                
                try:
                    function_to_call = getattr(functions, 'get_unit_details')
                    output = function_to_call({"unit_id": unit_id})
                    logging.info(f"✅ get_unit_details executed successfully for ID: {unit_id}")
                    
                    return {
                        "function_output": output,
                        "function_name": "get_unit_details"
                    }
                except Exception as e:
                    logging.error(f"🚫 Error executing get_unit_details: {e}")
                    return {
                        "error": f"Error executing get_unit_details: {str(e)}"
                    }
        # Pattern 1: _call:function_name(...)
        pattern1 = r'_call:(\w+)\(([^)]+)\)'
        match1 = re.search(pattern1, response_text)
        
        # Pattern 2: استدعي الأداة `function_name` بالمعلومات التالية:
        pattern2 = r'استدعي الأداة\s+`(\w+)`\s+بالمعلومات التالية:'
        match2 = re.search(pattern2, response_text)
        
        # Pattern 3: ```python function_name(...)```
        pattern3 = r'```python\s+(\w+)\(([^)]+)\)'
        match3 = re.search(pattern3, response_text)
        
        # Pattern 4: function_name(...) in text
        pattern4 = r'(\w+)\(([^)]+)\)'
        match4 = re.search(pattern4, response_text)
        
        function_name = None
        function_args_str = None
        
        if match1:
            function_name = match1.group(1)
            function_args_str = match1.group(2)
        elif match2:
            function_name = match2.group(1)
            # Extract arguments from the following lines
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if 'استدعي الأداة' in line:
                    # Look for arguments in subsequent lines
                    args_lines = lines[i+1:i+10]  # Check next 10 lines
                    function_args_str = '\n'.join(args_lines)
                    break
        elif match3:
            function_name = match3.group(1)
            function_args_str = match3.group(2)
        elif match4:
            function_name = match4.group(1)
            function_args_str = match4.group(2)
        
        if function_name and function_args_str:
            logging.info(f"🔧 Parsed function call: {function_name} with args: {function_args_str}")
            
            # Parse arguments
            function_args = {}
            
            # Try to parse key-value pairs
            if ':' in function_args_str:
                # Handle key: value format
                pairs = re.findall(r'`?(\w+)`?\s*:\s*`?"?([^`"\n,]+)`?"?', function_args_str)
                for key, value in pairs:
                    # Clean up the value
                    value = value.strip().strip('"').strip("'")
                    # Convert to appropriate type
                    if value.isdigit():
                        function_args[key] = int(value)
                    elif value.replace('.', '').isdigit():
                        function_args[key] = float(value)
                    else:
                        function_args[key] = value
            else:
                # Handle comma-separated format
                args = function_args_str.split(',')
                for arg in args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if value.isdigit():
                            function_args[key] = int(value)
                        elif value.replace('.', '').isdigit():
                            function_args[key] = float(value)
                        else:
                            function_args[key] = value
            
            # Map common parameter names
            if 'max_price' in function_args:
                function_args['budget'] = function_args.pop('max_price')
            if 'min_bedrooms' in function_args:
                function_args['bedrooms'] = function_args.pop('min_bedrooms')
            if 'max_bedrooms' in function_args:
                function_args['bedrooms'] = function_args.pop('max_bedrooms')
            if 'city' in function_args:
                function_args['location'] = function_args.pop('city')
            if 'exclude_ids' in function_args:
                function_args['excluded_ids'] = function_args.pop('exclude_ids')
            
            logging.info(f"🔧 Parsed arguments: {function_args}")
            
            # Execute the function
            try:
                function_to_call = getattr(functions, function_name)
                output = function_to_call(function_args)
                logging.info(f"✅ Function {function_name} executed successfully!")
                
                return {
                    "function_output": output,
                    "function_name": function_name
                }
                
            except Exception as e:
                logging.error(f"🚫 Error executing function {function_name}: {e}")
                return {
                    "error": f"Error executing function {function_name}: {str(e)}"
                }
        
        return None
        
    except Exception as e:
        logging.error(f"🚫 Error parsing function call: {e}")
        return None

def check_gemini_setup():
    """Check if Gemini API is properly configured"""
    try:
        api_key = getattr(variables, "GEMINI_API_KEY", None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in variables.py")
        
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
    """Get conversation context for a session using Gemini chat sessions"""
    try:
        # Get chat session from memory manager
        chat = memory_manager(session_id)
        
        # Get legacy context for backward compatibility
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                "user_preferences": {},
                "previous_messages": [],
                "current_search_results": [],
                "user_info": {}
            }
        
        # Combine Gemini chat session with legacy context
        context = conversation_memory[session_id].copy()
        
        # Note: Gemini chat sessions handle memory internally
        # We don't need to extract summary/entities manually
        context.update({
            "summary": "",  # Gemini handles this internally
            "entities": {}   # Gemini handles this internally
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
    """Update conversation context using Gemini chat sessions"""
    try:
        if key == "current_search_results":
            memory_manager.update_search_results(session_id, value)
        else:
            # Fallback to legacy memory
            if session_id not in conversation_memory:
                conversation_memory[session_id] = {
                    "user_preferences": {},
                    "previous_messages": [],
                    "current_search_results": [],
                    "user_info": {}
                }
            conversation_memory[session_id][key] = value
    except Exception as e:
        logging.error(f"Error updating conversation context: {e}")
        # Fallback to legacy memory only
        if session_id not in conversation_memory:
            conversation_memory[session_id] = {
                "user_preferences": {},
                "previous_messages": [],
                "current_search_results": [],
                "user_info": {}
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
    """Process Gemini response using built-in chat sessions with memory and automatic function calling"""
    try:
        # Get conversation context using Gemini chat sessions
        context = get_conversation_context(session_id)
        
        # Extract new preferences from current message
        new_preferences = extract_user_preferences_from_message(user_message)
        
        # Get updated context after preference update
        context = get_conversation_context(session_id)
        
        # Create enhanced context-aware prompt with Gemini chat sessions
        context_prompt = ""
        
        # Add current preferences
        if context.get("preferences", {}):
            context_prompt += f"\n### تفضيلات المستخدم الحالية:\n"
            for key, value in context["preferences"].items():
                context_prompt += f"- {key}: {value}\n"
        
        # Create the full prompt with system instructions and enhanced context
        system_prompt = f"{config.assistant_instructions}\n\n### Examples:\n{config.examples}"
        full_prompt = f"{system_prompt}\n\n{context_prompt}\n\n### رسالة المستخدم:\n{user_message}"
        
        # Get chat session for this session_id (pre-configured with tools)
        chat = memory_manager(session_id)
        
        # Send message to chat session (this automatically handles memory)
        logging.info(f"Generating response for user message: {user_message[:100]}...")
        response = chat.send_message(full_prompt)
        
        # Check if response contains native function calls (Gemini tool calling)
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
                                client_info = (
                                    config.client_sessions.get(session_id, {})
                                    or get_session(session_id)
                                    or {}
                                )
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
                                    client_info = (
                                        config.client_sessions.get(session_id, {})
                                        or get_session(session_id)
                                        or {}
                                    )
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
                                    client_info = (
                                        config.client_sessions.get(session_id, {})
                                        or get_session(session_id)
                                        or {}
                                    )
                                    user_id = client_info.get("user_id")
                                    if user_id:
                                        prefs = functions.get_conversation_preferences(session_id, user_id)
                                        if prefs.get("location") and not function_args.get("location"):
                                            function_args["location"] = prefs.get("location")
                                        if prefs.get("property_type") and not function_args.get("property_type"):
                                            function_args["property_type"] = prefs.get("property_type")
                                except Exception as _e:
                                    logging.warning(f"Could not augment insight_search with context: {_e}")

                            # Execute tool function with parsed args
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
        
        # Fallback: Try to parse function calls from text response
        # Note: We prefer native tool-calling; this is a safety net only
        function_result = parse_function_call_from_text(response_text)
        if function_result:
            logging.info(f"🔧 Function call parsed from text: {function_result.get('function_name')}")
            return function_result
        
        # Also check the original user message for unit detail requests
        user_function_result = parse_function_call_from_text(user_message)
        if user_function_result:
            logging.info(f"🔧 Function call parsed from user message: {user_function_result.get('function_name')}")
            return user_function_result
        
        # Check if we should trigger a search based on conversation context
        if session_id:
            try:
                client_info = (
                    config.client_sessions.get(session_id, {})
                    or get_session(session_id)
                    or {}
                )
                user_id = client_info.get("user_id")
                if user_id:
                    prefs = functions.get_conversation_preferences(session_id, user_id)
                    
                    # Check if we have the minimum required information for a search
                    has_location = prefs.get("location") or "القاهرة الجديدة" in user_message or "العاصمة" in user_message
                    has_budget = prefs.get("budget", 0) > 0
                    has_property_type = prefs.get("property_type") or "شقة" in user_message or "فيلا" in user_message
                    
                    # If we have enough info and user seems ready for search, trigger it
                    if has_location and has_budget and has_property_type:
                        logging.info(f"🔍 Auto-triggering search with preferences: {prefs}")
                        
                        # Determine if it should be new launch or existing unit search
                        query_type = functions.classify_query_type_with_llm(user_message)
                        
                        if query_type == "new_launch":
                            function_args = {
                                "property_type": prefs.get("property_type", "شقة"),
                                "location": prefs.get("location", "القاهرة الجديدة"),
                                "compound": prefs.get("compound_name", ""),
                                "session_id": session_id
                            }
                            output = functions.search_new_launches(function_args)
                            return {
                                "function_output": output,
                                "function_name": "search_new_launches"
                            }
                        else:
                            # Default to existing units search
                            function_args = {
                                "location": prefs.get("location", "القاهرة الجديدة"),
                                "budget": prefs.get("budget", 0),
                                "property_type": prefs.get("property_type", "شقة"),
                                "bedrooms": prefs.get("bedrooms", 0),
                                "bathrooms": prefs.get("bathrooms", 0),
                                "compound": prefs.get("compound_name", ""),
                                "session_id": session_id
                            }
                            output = functions.property_search(function_args)
                            return {
                                "function_output": output,
                                "function_name": "property_search"
                            }
            except Exception as e:
                logging.warning(f"Could not auto-trigger search: {e}")
        
        logging.info(f"Gemini text response: {response_text[:200]}...")
        
        # Note: Memory is automatically saved by the chat session
        # No need to manually save context - Gemini handles it internally
        
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
