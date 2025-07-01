# CardioPredict Deployment Guide

## Free Hosting Options

### 1. **Render** (Recommended) ⭐
- **URL**: https://render.com
- **Free Tier**: 750 hours/month, sleeps after 15 min inactivity
- **Setup**:
  1. Push code to GitHub
  2. Connect GitHub repo to Render
  3. Use `render.yaml` configuration (already created)
  4. Deploy automatically

### 2. **Railway** ⭐
- **URL**: https://railway.app
- **Free Tier**: $5 credit monthly, usage-based
- **Setup**:
  1. Connect GitHub repo
  2. Select Flask template
  3. Uses `Procfile` (already created)
  4. Auto-deploys on git push

### 3. **Fly.io**
- **URL**: https://fly.io
- **Free Tier**: 3 shared-cpu VMs, 160GB outbound transfer
- **Setup**:
  ```bash
  flyctl launch
  flyctl deploy
  ```

### 4. **Heroku** (Limited Free Tier)
- **URL**: https://heroku.com
- **Note**: Free tier discontinued, but still affordable
- **Setup**:
  ```bash
  heroku create cardiopredict-app
  git push heroku main
  ```

## Quick Start Deployment

### Option A: Render (Easiest)
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/CardioPredict.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to https://render.com
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Render will auto-detect the `render.yaml` configuration
   - Click "Deploy"

### Option B: Railway
1. **Push to GitHub** (same as above)
2. **Deploy on Railway**:
   - Go to https://railway.app
   - Login with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your CardioPredict repo
   - Railway auto-detects Flask and uses `Procfile`

## Environment Configuration

### Required Environment Variables:
- `FLASK_ENV=production`
- `SECRET_KEY=your-secret-key-here`
- `PORT=5000` (auto-set by most platforms)

### Optional (for database):
- `DATABASE_URL=postgresql://...` (for PostgreSQL)

## Local Testing Before Deployment

```bash
# Test production build locally
cd web_platform
pip install -r requirements.txt
export FLASK_ENV=production
export SECRET_KEY=test-secret-key
gunicorn --bind 0.0.0.0:5000 app:app
```

Visit: http://localhost:5000

## Database Notes

- **Development**: Uses SQLite (included in repo)
- **Production**: Can use PostgreSQL (Render/Railway provide free tiers)
- Database will be automatically created on first run

## Domain Setup (Optional)

Most platforms provide:
- **Render**: `your-app-name.onrender.com`
- **Railway**: `your-app-name.up.railway.app`
- **Fly.io**: `your-app-name.fly.dev`

You can later connect a custom domain.

## Monitoring

Your deployed app includes:
- Health check endpoint: `/`
- API stats: `/api/stats`
- Error logging built-in

## Troubleshooting

### Common Issues:
1. **Build fails**: Check requirements.txt paths
2. **App won't start**: Check PORT environment variable
3. **Database errors**: Ensure database URL is correct

### Logs:
- **Render**: View in dashboard
- **Railway**: `railway logs`
- **Fly.io**: `flyctl logs`

## Security

For production:
1. Set strong SECRET_KEY
2. Use environment variables for sensitive data
3. Enable HTTPS (auto-enabled on most platforms)
4. Consider adding rate limiting

---

**Estimated Setup Time**: 5-10 minutes for basic deployment
**Cost**: Free tier limits vary, all platforms offer substantial free usage
