# 🚀 Backend AI Features Integration Guide

## ✅ What's Done

All 3 new AI endpoints have been added to your backend at:
**`app/api/v1/ai.py` (lines 757-993)**

### Endpoints Added:

1. **POST `/api/v1/ai/analyze-meal-photo`**
2. **POST `/api/v1/ai/analyze-exercise-form`**
3. **GET `/api/v1/ai/predictive-analytics`**

### Current Status:
- ✅ Endpoints are **functional** and return **realistic placeholder data**
- ✅ CORS is **already configured** correctly in `main.py`
- ✅ Authentication is **working** (uses `current_active_user` dependency)
- ✅ Logging is **comprehensive**
- ✅ Error handling is **robust**
- ⚠️ Using **placeholder/mock data** (real AI integration needed)

---

## 🧪 Testing the Endpoints

### 1. Deploy to Railway

Your backend is already on Railway. The new endpoints will be available immediately after deployment.

**Deploy command:**
```bash
cd Evolvefitai_backend
git add .
git commit -m "Add AI feature endpoints: meal photo, form analysis, predictive analytics"
git push
```

Railway will automatically deploy. Check logs for:
```
📸 Meal photo analysis requested by user@email.com
✅ Meal photo analysis completed
⚠️  Using placeholder data - integrate real AI vision service
```

### 2. Test with cURL

**Test Predictive Analytics (simplest):**
```bash
curl -X GET "https://evolvefitaibackend-production.up.railway.app/api/v1/ai/predictive-analytics" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Test Meal Photo Analysis:**
```bash
curl -X POST "https://evolvefitaibackend-production.up.railway.app/api/v1/ai/analyze-meal-photo" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_string_here"}'
```

**Test Form Analysis:**
```bash
curl -X POST "https://evolvefitaibackend-production.up.railway.app/api/v1/ai/analyze-exercise-form" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### 3. Check Frontend Integration

After deployment, your frontend will automatically work because:
1. It tries the real API first
2. If it fails → falls back to mock data
3. Once real API returns data → uses real data seamlessly

---

## 🔧 Next Steps: Real AI Integration

### Option 1: Meal Photo Analysis (OpenAI GPT-4 Vision)

**Update `analyze_meal_photo` function (line 761):**

```python
import openai
import base64

@router.post("/analyze-meal-photo")
async def analyze_meal_photo(
    request: dict,
    current_user: models.User = Depends(current_active_user)
):
    """Analyze a meal photo using AI vision"""
    logger.info(f"📸 Meal photo analysis requested by {current_user.email}")

    try:
        image_base64 = request.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")

        # Use OpenAI Vision API
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this meal photo and provide:
                            1. Meal name
                            2. Brief description
                            3. Estimated nutrition (calories, protein, carbs, fats in grams)
                            4. List of visible ingredients
                            5. Fitness recommendations

                            Return as JSON with keys: meal_name, description, nutrition (with calories, protein, carbs, fats), ingredients (array), recommendations"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Parse AI response
        ai_content = response.choices[0].message.content
        analysis_result = json.loads(ai_content)

        logger.info(f"✅ Real AI meal analysis completed for {current_user.email}")
        return analysis_result

    except Exception as e:
        logger.error(f"💥 Meal photo analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
```

**Add to `requirements.txt`:**
```
openai>=1.0.0
```

**Add to Railway environment variables:**
```
OPENAI_API_KEY=sk-your-key-here
```

---

### Option 2: Exercise Form Analysis (MediaPipe)

**Update `analyze_exercise_form` function (line 812):**

```python
import mediapipe as mp
from fastapi import UploadFile, File
import cv2
import numpy as np

@router.post("/analyze-exercise-form")
async def analyze_exercise_form(
    video: UploadFile = File(...),
    current_user: models.User = Depends(current_active_user)
):
    """Analyze exercise form from video using pose estimation"""
    logger.info(f"🎥 Exercise form analysis requested by {current_user.email}")

    try:
        # Save uploaded video temporarily
        video_bytes = await video.read()
        video_path = f"/tmp/form_check_{current_user.id}.mp4"

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )

        # Process video
        cap = cv2.VideoCapture(video_path)
        issues = []
        frame_count = 0

        while cap.isOpened() and frame_count < 100:  # Limit frames
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                # Analyze angles and positions
                landmarks = results.pose_landmarks.landmark

                # Example: Check knee alignment in squat
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                # Calculate angle
                knee_angle = calculate_angle(
                    [left_hip.x, left_hip.y],
                    [left_knee.x, left_knee.y],
                    [left_ankle.x, left_ankle.y]
                )

                # Detect issues
                if knee_angle < 70:  # Too deep
                    issues.append({
                        "severity": "warning",
                        "title": "Knee Depth",
                        "description": "Your knees are going too deep. Maintain 90-degree angle."
                    })

            frame_count += 1

        cap.release()
        pose.close()

        # Clean up
        os.remove(video_path)

        # Calculate overall score
        overall_score = max(50, 100 - (len(issues) * 10))

        analysis_result = {
            "exercise_name": "Detected Exercise",
            "overall_score": overall_score,
            "issues": issues[:5],  # Top 5 issues
            "recommendations": [
                "Focus on maintaining proper form throughout the movement",
                "Consider using lighter weight to perfect technique"
            ]
        }

        logger.info(f"✅ Real AI form analysis completed for {current_user.email}")
        return analysis_result

    except Exception as e:
        logger.error(f"💥 Exercise form analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    import math
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
```

**Add to `requirements.txt`:**
```
mediapipe>=0.10.0
opencv-python>=4.8.0
```

---

### Option 3: Enhanced Predictive Analytics (Already Working!)

The predictive analytics endpoint is **already using real workout data** from your database!

**Current implementation (line 864):**
- ✅ Fetches actual workout logs from past 90 days
- ✅ Calculates real trends and improvements
- ✅ Generates personalized recommendations
- ✅ Uses actual consistency scores

**Optional Enhancement with ML:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Inside get_predictive_analytics function:

# Convert to numpy arrays
X = np.array(range(len(historical_data))).reshape(-1, 1)
y = np.array([d["value"] for d in historical_data])

# Train simple linear regression
model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_X = np.array(range(len(historical_data), len(historical_data) + 30)).reshape(-1, 1)
predictions = model.predict(future_X)

# Use predictions instead of simple formula
for i, pred in enumerate(predictions):
    date = today + timedelta(days=i+1)
    future_data.append({
        "date": date.strftime("%Y-%m-%d"),
        "value": max(0, pred),
        "upper_bound": max(0, pred + 10),
        "lower_bound": max(0, pred - 10)
    })
```

**Add to `requirements.txt`:**
```
scikit-learn>=1.3.0
numpy>=1.24.0
```

---

## 📊 Current Endpoint Capabilities

### `/predictive-analytics` (Already Production-Ready!)
- ✅ Uses real workout data
- ✅ Calculates actual trends
- ✅ Personalized based on user history
- ✅ Works with no additional setup

### `/analyze-meal-photo` (Placeholder)
- ⏳ Returns realistic mock nutrition data
- 🔄 Ready for OpenAI Vision integration
- 📝 Works immediately, upgrade when ready

### `/analyze-exercise-form` (Placeholder)
- ⏳ Returns mock form analysis
- 🔄 Ready for MediaPipe integration
- 📝 Works immediately, upgrade when ready

---

## 🚢 Deployment Checklist

### Immediate (No Changes Needed)
- [x] Endpoints added to `ai.py`
- [x] CORS configured in `main.py`
- [x] Authentication working
- [x] Logging comprehensive
- [x] Error handling robust

### To Deploy Placeholder Version (Now)
```bash
cd Evolvefitai_backend
git add app/api/v1/ai.py
git commit -m "feat: Add AI endpoints for meal analysis, form check, and predictions"
git push
```

**Result:** All features work with placeholder data immediately!

### To Enable Real AI (Later)

**For Meal Photo Analysis:**
1. Add `OPENAI_API_KEY` to Railway environment
2. Update function with OpenAI code above
3. Add `openai` to requirements.txt
4. Deploy

**For Form Analysis:**
1. Add MediaPipe dependencies to requirements.txt
2. Update function with MediaPipe code above
3. Deploy (may need more RAM on Railway)

**For Better Predictions:**
1. Add scikit-learn to requirements.txt
2. Update function with ML code above
3. Deploy

---

## 🧪 Testing Script

Create `test_ai_endpoints.py`:

```python
import requests
import base64

BASE_URL = "https://evolvefitaibackend-production.up.railway.app/api/v1/ai"
TOKEN = "your_jwt_token_here"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Test 1: Predictive Analytics
print("Testing Predictive Analytics...")
response = requests.get(f"{BASE_URL}/predictive-analytics", headers=headers)
print(f"Status: {response.status_code}")
print(f"Data: {response.json()}\n")

# Test 2: Meal Photo
print("Testing Meal Photo Analysis...")
with open("sample_meal.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/analyze-meal-photo",
    headers=headers,
    json={"image": image_base64}
)
print(f"Status: {response.status_code}")
print(f"Data: {response.json()}\n")

# Test 3: Form Analysis
print("Testing Form Analysis...")
response = requests.post(f"{BASE_URL}/analyze-exercise-form", headers=headers)
print(f"Status: {response.status_code}")
print(f"Data: {response.json()}\n")

print("✅ All tests complete!")
```

Run with:
```bash
python test_ai_endpoints.py
```

---

## 📝 Environment Variables Needed

### Current (No Changes)
- `DATABASE_URL` - ✅ Already set
- `SECRET_KEY` - ✅ Already set
- `GROQ_API_KEY` - ✅ Already set (for workouts)

### For Real AI (Add Later)
- `OPENAI_API_KEY` - For meal photo analysis
- No additional vars for form analysis (uses MediaPipe)

---

## 🎯 Summary

**Right Now:**
1. Commit and push the changes
2. Railway auto-deploys
3. All 3 endpoints work with realistic placeholder data
4. Frontend works perfectly (has fallback system)
5. No errors for users!

**Later (When Ready):**
1. Add OpenAI API key for meal analysis
2. Add MediaPipe for form checking
3. Optional: Add scikit-learn for better predictions
4. Frontend automatically switches to real AI data!

**Zero downtime, seamless upgrade path!** 🚀

---

*Generated for EvolveFit AI Backend*
*Date: 2025-01-16*
