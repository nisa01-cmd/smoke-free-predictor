# 🚀 Deployment Guide - Smoke-Free Predictor

This guide provides step-by-step instructions for deploying your Smoke-Free Predictor web application to various cloud platforms.

## 📋 Prerequisites

- Git installed on your machine
- Your project files ready (you already have these!)
- A cloud platform account (we'll cover multiple options)

## 🎯 Quick Deploy Options

### Option 1: Heroku (Recommended - Easy & Free Tier)

#### Step 1: Create Heroku Account
1. Go to [heroku.com](https://heroku.com)
2. Sign up for a free account
3. Install Heroku CLI: [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

#### Step 2: Prepare Your Repository
```bash
# Make sure you're in your project directory
cd smoke-free-predictor

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit for deployment"
```

#### Step 3: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create a new Heroku app (replace 'your-app-name' with a unique name)
heroku create your-smoke-free-predictor

# Deploy your code
git push heroku main

# Open your deployed app
heroku open
```

#### Step 4: Configure Environment Variables (Optional)
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-super-secret-key-here
```

**Your app will be live at: `https://your-app-name.herokuapp.com`**

---

### Option 2: Railway (Modern & Fast)

#### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

#### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Connect your GitHub account and select your repository
4. Railway will automatically detect it's a Python app and deploy!

**Your app will be live at: `https://your-project.up.railway.app`**

---

### Option 3: Render (Free Tier Available)

#### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

#### Step 2: Deploy
1. Click "New Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Python Version:** 3.9.18

**Your app will be live at: `https://your-app-name.onrender.com`**

---

### Option 4: Vercel (Serverless)

#### Step 1: Install Vercel CLI
```bash
npm i -g vercel
```

#### Step 2: Create vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "./app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/"
    }
  ]
}
```

#### Step 3: Deploy
```bash
vercel --prod
```

---

## 🔧 Configuration for Production

### Environment Variables
Set these in your deployment platform:

```bash
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-change-this
PORT=5000  # Usually auto-set by platforms
```

### Database Configuration (if needed)
The app currently uses in-memory models, but you can configure:
```bash
DATABASE_URL=your-database-url-here
MODEL_STORAGE_PATH=/app/models/
```

## 🔍 Testing Your Deployment

After deployment, test these endpoints:

1. **Home Page:** `https://your-app.com/`
2. **Health Check:** `https://your-app.com/api/health`
3. **Prediction Form:** `https://your-app.com/predict`
4. **API Test:**
   ```bash
   curl -X POST https://your-app.com/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "gender": "Female",
       "stress_level": 6,
       "peer_pressure": 4,
       "exercise_freq": 3,
       "sleep_hours": 7.5,
       "motivation_score": 8.2,
       "support_system": "high"
     }'
   ```

## 🛠️ Troubleshooting

### Common Issues:

#### 1. App Crashes on Startup
- Check logs: `heroku logs --tail` (for Heroku)
- Ensure all dependencies are in requirements.txt
- Verify Python version in runtime.txt

#### 2. Model Not Loading
- Check if model files are included in git
- Ensure model path is correct
- The app creates a demo model if none exists

#### 3. Static Files Not Loading
- Ensure templates/ and static/ directories are committed
- Check Flask static file configuration

#### 4. Memory Issues
- Reduce model complexity if needed
- Use lighter dependencies
- Consider upgrading to paid tier for more memory

## 📊 Monitoring & Analytics

### Add Application Monitoring:
1. **Heroku:** Add New Relic or Papertrail
2. **Railway:** Built-in metrics
3. **Render:** Built-in monitoring
4. **Vercel:** Analytics dashboard

### Health Checks:
All deployments include `/api/health` endpoint for monitoring.

## 🔒 Security Considerations

### For Production Use:
1. **Change SECRET_KEY** in environment variables
2. **Enable HTTPS** (most platforms do this automatically)
3. **Rate Limiting:** Consider adding Flask-Limiter
4. **Input Validation:** Already included in the app
5. **CORS:** Configure if needed for API access

## 💰 Cost Estimates

### Free Tiers:
- **Heroku:** 550 hours/month free (sleeps after 30min idle)
- **Railway:** $5 credit monthly (enough for small apps)
- **Render:** 750 hours/month free
- **Vercel:** Generous free tier

### Paid Plans (if needed):
- **Heroku Hobby:** $7/month
- **Railway:** $5+ per month
- **Render:** $7/month
- **Vercel Pro:** $20/month

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live web application accessible via URL
- ✅ REST API for programmatic access
- ✅ Professional UI with responsive design
- ✅ File upload for batch predictions
- ✅ Real-time predictions with AI model

## 📞 Support

If you encounter issues:
1. Check the platform-specific documentation
2. Review application logs
3. Test locally first: `python app.py`
4. Ensure all files are committed to git

## 🔄 Updates & Maintenance

To update your deployed app:
```bash
# Make changes to your code
git add .
git commit -m "Update description"

# For Heroku:
git push heroku main

# For other platforms:
git push origin main  # They auto-deploy from main branch
```

---

**🎊 Congratulations! Your Smoke-Free Predictor is now live and accessible to anyone worldwide!**

Share your URL and start helping people with AI-powered smoking cessation predictions!