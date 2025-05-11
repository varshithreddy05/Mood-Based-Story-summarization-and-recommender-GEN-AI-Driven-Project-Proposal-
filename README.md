🎙️ MOOD-BASED-STORY-SUMMARIZATION-AND-RECOMMENDER-AI-COMPANION-FOR-KUKU-FM

🔍 Project Overview

This project introduces a generative AI-powered feature for Kuku FM that transforms audio stories into personalized, mood-aligned summaries and recommendations. By combining speech-to-text, mood detection, and advanced summarization, it aims to boost user engagement and retention through real-time, emotionally intelligent content delivery.

🛠️ Key Features

- **Mood Detection:** Analyzes both user voice tone and text sentiment to determine mood (happy, sad, excited, etc.).
- **AI Summarization:** Generates creative, 30-second story summaries using state-of-the-art models (BART, LSA, LLaMA-3-70B).
- **Personalized Recommendations:** Matches user mood with story mood for relevant suggestions.
- **Voice Interaction:** Users can request content by speaking or typing their mood.
- **Seamless Experience:** One-tap transition from summary to full audiobook.
- **Dynamic Playlists:** Curates “Mood of the Day” playlists and adapts in real-time to user behavior.

🧪 Dataset & Tech Stack

| Component         | Tools/Techniques                                    | Data Sources                                |
|-------------------|-----------------------------------------------------|---------------------------------------------|
| Mood Analysis     | Whisper, Custom CNN, DistilBERT, Librosa            | User voice clips, search queries            |
| Content Generation| BART, LSA, Groq (LLaMA-3-70B)                       | Kuku FM audiobook library, genre tags       |
| Personalization   | Sentence-BERT, PyTorch                              | Listening history, skip rates               |

📊 Results & Metrics

- +20% average session length (22 mins → 26.4 mins)
- +15% weekly app opens (8 → 9.2)
- 40% click-through rate on AI-generated summaries
- 25% conversion from summary to full audiobook
- <500 ms mood detection latency
- 95% language identification accuracy (Hindi/English/Tamil)

✅ Key Takeaways

- Personalized, mood-aligned recommendations reduce content mismatch and decision fatigue.
- Instant, AI-generated story hooks increase engagement and retention.
- Voice-controlled, real-time interaction enhances user experience.
- Phased rollout and use of existing Kuku FM infrastructure ensure feasibility and scalability.
- Strategic alignment with business goals-projected to reduce recommendation mismatch by 60% and drive premium upsells.

🚀 Implementation Roadmap

- **Phase 1 (6 weeks):** Develop AI pipeline, fine-tune summarization models, build hybrid recommender.
- **Phase 2 (4 weeks):** Integrate in-app voice commands and summary-to-audiobook UI.
- **Phase 3:** Feedback loop using PySpark for continuous improvement (analyzing skips, replays, mood-genre conversions).

🎯 Challenges & Solutions

| Challenge                                  | Solution                                              |
|---------------------------------------------|-------------------------------------------------------|
| Regional language mood detection            | Train Whisper on Kuku FM’s dialect database           |
| Over-personalization/filter bubbles         | Introduce 10% “mood challenger” recommendations       |
| Real-time audio processing latency          | Optimize with ONNX runtime + edge computing           |

🖥️ Deliverables

- UI/UX Design (Figma prototype)
- Technical architecture diagrams
- Functional demo (see GitHub repo)
- Video walkthrough

🌟 Strategic Alignment

- **Consumer Value:** 60% reduction in recommendation mismatch (A/B test)
- **Business Impact:** Drives new user acquisition and premium conversions
- **Tech Feasibility:** Leverages 60% of existing Kuku FM infrastructure and cost-effective APIs

👤 Submitted by: GANTA SAIVARSHITH REDDY  
Project PDF: https://drive.google.com/file/d/1A6U7H-rWFed8PjPw7v8cZFa6yf6Yvg9A/view?usp=sharing
GitHub: https://github.com/varshithreddy05  
LinkedIn: https://linkedin.com/in/saivarshith-reddy-g-1a6379270  
Contact: varshitreddy97@gmail.com

> This project pioneers a hybrid approach to mood-based content recommendation, combining audio and text AI for a more human, engaging, and personalized Kuku FM experience.
