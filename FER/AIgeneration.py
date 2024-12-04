import os
import sys
from datetime import datetime
import pandas as pd
import google.generativeai as genai
from typing import Dict, Optional

class AIInterviewEmotionReportGenerator:
    def __init__(self, csv_path: Optional[str] = None, txt_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the report generator with optional CSV and text paths
        
        :param csv_path: Path to the emotion detection CSV file
        :param api_key: Google AI API key for Gemini
        """
        # Validate and find files if not provided
        self.csv_path = self._find_latest_file(csv_path, 'emotion_report_', '.csv')
        
        if not self.csv_path:
            raise FileNotFoundError("Could not find required CSV. Please provide explicit paths.")
        
        # Rest of the initialization remains the same as in the original code
        self._initialize_data_and_model(api_key)
    
    def _find_latest_file(self, provided_path: Optional[str], prefix: str, extension: str) -> Optional[str]:
        """
        Find the latest file matching criteria
        
        :param provided_path: Explicitly provided file path
        :param prefix: Filename prefix to search for
        :param extension: File extension to match
        :return: Path to the most recent file or None
        """
        # If path is explicitly provided and exists, return it
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # Search directories: script directory, current directory, parent directory
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),  # Script's directory
            os.getcwd(),  # Current working directory
            os.path.dirname(os.getcwd())  # Parent directory
        ]
        
        matching_files = []
        for directory in search_dirs:
            try:
                # Find files in the directory matching prefix and extension
                files = [
                    os.path.join(directory, f) 
                    for f in os.listdir(directory) 
                    if f.startswith(prefix) and f.endswith(extension)
                ]
                matching_files.extend(files)
            except Exception:
                continue
        
        # If no matching files found, return None
        if not matching_files:
            print(f"No files found with prefix '{prefix}' and extension '{extension}'")
            return None
        
        # Return the most recently created file
        return max(matching_files, key=os.path.getctime)
    
    def _initialize_data_and_model(self, api_key: Optional[str]):
        """
        Initialize data and AI model
        
        :param api_key: Google AI API key
        """
        # API key configuration 
        if api_key:
            genai.configure(api_key=api_key)
        elif 'GOOGLE_API_KEY' in os.environ:
            genai.configure(api_key='AIzaSyCr2qgNNlzJATGoaeuDSNt3uEXbncPt500')
        else:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass key directly.")
        
        # Load CSV data
        self.df = pd.read_csv(self.csv_path)
        
        # Convert time columns to datetime
        self.df['Start Time'] = pd.to_datetime(self.df['Start Time'])
        self.df['End Time'] = pd.to_datetime(self.df['End Time'])
        
        # Set up Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
    
    def _map_emotions_to_interview_context(self, emotion: str, duration_percentage: float) -> str:
        """
        Map basic emotions to interview-specific emotional contexts with contextual interpretation
        
        :param emotion: Original emotion detected
        :param duration_percentage: Percentage of time spent in this emotion
        :return: Interview-specific emotional interpretation
        """
        emotion_mapping = {
            'happy': {
                'base': 'Confident Composure',
                'high_duration': 'Genuine Enthusiasm',
                'low_duration': 'Forced Positivity'
            },
            'sad': {
                'base': 'Reflective Contemplation',
                'high_duration': 'Emotional Vulnerability',
                'low_duration': 'Momentary Self-Doubt'
            },
            'fear': {
                'base': 'Performance Anxiety',
                'high_duration': 'Overwhelming Nervousness',
                'low_duration': 'Manageable Tension'
            },
            'surprise': {
                'base': 'Adaptive Responsiveness',
                'high_duration': 'Quick Thinking',
                'low_duration': 'Momentary Confusion'
            },
            'angry': {
                'base': 'Passionate Conviction',
                'high_duration': 'Defensive Stance',
                'low_duration': 'Momentary Frustration'
            },
            'neutral': {
                'base': 'Clueless Demeanor',
                'high_duration': 'Complete Disengagement',
                'low_duration': 'Professional Restraint'
            }
        }
        
        # Determine the appropriate description based on duration
        emotion_lower = emotion.lower()
        if emotion_lower not in emotion_mapping:
            return emotion
        
        # Categorize duration
        if duration_percentage > 50:
            return emotion_mapping[emotion_lower]['high_duration']
        elif duration_percentage > 20:
            return emotion_mapping[emotion_lower]['base']
        else:
            return emotion_mapping[emotion_lower]['low_duration']
    
    def analyze_emotion_distribution(self) -> Dict[str, float]:
        """
        Calculate the distribution of emotions with interview-specific mapping
        
        :return: Dictionary of emotion percentages
        """
        # Calculate emotion durations
        emotion_durations = self.df.groupby('Emotion').apply(
            lambda x: (x['End Time'] - x['Start Time']).sum().total_seconds()
        )
        total_duration = emotion_durations.sum()
        
        # Map and calculate percentages with context-aware mapping
        emotion_percentages = {}
        for emotion, duration in emotion_durations.items():
            percentage = (duration / total_duration * 100)
            mapped_emotion = self._map_emotions_to_interview_context(emotion, percentage)
            emotion_percentages[mapped_emotion] = percentage
        
        return emotion_percentages
    
    def generate_ai_interview_analysis(self) -> str:
        """
        Generate an AI-powered, personalized interview emotion and Q&A analysis
        
        :return: Detailed narrative report from Gemini
        """
        # Prepare input data for AI analysis
        emotion_dist = self.analyze_emotion_distribution()
        
        # Format emotion distribution for AI input
        emotion_summary = "\n".join([
            f"{emotion}: {percentage:.2f}%" 
            for emotion, percentage in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)
        ])
        
        # Construct prompt for Gemini with personalized approach
        prompt = f"""Hey there! Let's dive into your interview performance with a friendly, supportive breakdown. 

Emotion Distribution:
{emotion_summary}

I'll help you understand how you came across during the interview and give you some super practical tips to shine even brighter next time!

🎯 Interview Emotion Insights:
Let's decode what your emotions were telling us during the interview. Here's a friendly breakdown:
- Confident Moments: When did you feel most self-assured and why?
- Reflective Pauses: Those thoughtful moments that show your depth
- Nervous Spots: No worries! We'll turn those butterflies into superpowers
- Adaptive Wins: Times you showed flexibility and quick thinking
- Passionate Points: Where your true excitement and conviction shined through

🚀 Personalized Performance Breakdown:
I want to give you a real, helpful analysis that feels like advice from a supportive friend. Let's cover:
1. Emotional Journey: How your feelings evolved during the interview
2. Communication Style: Your unique way of expressing yourself
3. Strengths Spotlight: What absolutely rocked in your interview
4. Growth Opportunities: Gentle, constructive tips to level up your interview game

🤔 Special Considerations:
- If you seemed neutral or hesitant for long periods, we'll explore why and how to boost your confidence
- Looking beyond just the emotions to your overall interview vibe
- Providing context-rich, actionable advice you can actually use

Key Focus Areas:
- Technical Knowledge Presentation
- Communication Clarity
- Emotional Intelligence
- Problem-Solving Approach
- Overall Interview Presence

Tone: Think of this as a chat with a supportive mentor who's genuinely excited to help you grow. Casual, encouraging, and packed with real-world advice you can implement in your next interview.

Bonus Challenge: After reading this, I want you to feel motivated, not discouraged. This is about your journey of continuous improvement! 💪🌟"""
        
        # Generate report using Gemini
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback to rule-based analysis if AI generation fails
            return self._generate_fallback_analysis(emotion_dist)
    
    def _generate_fallback_analysis(self, emotion_dist: Dict[str, float]) -> str:
        """
        Generate a fallback analysis if AI generation fails
        
        :param emotion_dist: Emotion distribution percentages
        :return: Fallback analysis report
        """
        fallback_report = "Personalized Interview Analysis\n\n"
        
        fallback_report += "Emotional Landscape:\n"
        fallback_report += "\n".join([
            f"{emotion}: {percentage:.2f}%" 
            for emotion, percentage in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fallback_report += "\n\nFallback Analysis:\n"
        fallback_report += "Unable to generate AI-powered analysis. "
        fallback_report += "Please refer to the emotional distribution for insights."
        
        return fallback_report
    
    def save_report(self, output_path: Optional[str] = None):
        """
        Save the generated report to a file
        
        :param output_path: Optional custom output path
        """
        if not output_path:
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_path = f'comprehensive_interview_analysis_{timestamp}.txt'
        
        # Generate AI-powered report
        summary = self.generate_ai_interview_analysis()
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Comprehensive Interview Analysis Report saved to {output_path}")
        return output_path

def main():
    # Optional: Specify paths explicitly
    csv_path = None  # or provide a specific path
    
    # API Key (replace with your actual key or set as environment variable)
    API_KEY = 'AIzaSyCr2qgNNlzJATGoaeuDSNt3uEXbncPt500'
    
    try:
        # Generate and save report
        report_generator = AIInterviewEmotionReportGenerator(
            csv_path=csv_path, 
            api_key=API_KEY
        )
        report_generator.save_report()
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()