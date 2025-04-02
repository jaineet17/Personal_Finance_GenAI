# Frontend Static Build and Deployment

This document outlines the process for building and deploying the static frontend assets for the Finance RAG application.

## Technology Stack

- **Framework**: React with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Deployment**: Vercel/Netlify (free tier)

## Prerequisites

Before proceeding with the build and deployment process, ensure you have:

1. Node.js 16+ installed
2. npm or yarn installed
3. A Vercel or Netlify account
4. API endpoint URL for the backend

## Local Build Process

### Environment Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Create a `.env.production` file with the following content:
   ```
   VITE_API_URL=https://your-api-endpoint.com/api
   VITE_ENABLE_ANALYTICS=false
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

### Building the Frontend

1. Run the production build command:
   ```bash
   npm run build
   ```

2. The build output will be generated in the `dist` directory.

3. Test the production build locally:
   ```bash
   npm run preview
   ```

### Optimizations Applied

The build process automatically applies several optimizations:

1. **JavaScript Bundling**:
   - Code splitting for route-based chunks
   - Tree shaking to remove unused code
   - Minification and compression

2. **CSS Optimization**:
   - Purging unused CSS with Tailwind's JIT compiler
   - CSS minification
   - Critical CSS extraction

3. **Asset Optimization**:
   - Image compression
   - Font subsetting
   - SVG optimization

4. **Performance Optimizations**:
   - Preloading of critical resources
   - Lazy loading of non-critical components
   - Module/nomodule pattern for better browser support

## Deployment Process

### Deploying to Vercel

1. Install Vercel CLI (optional but recommended):
   ```bash
   npm install -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy the application:
   ```bash
   vercel --prod
   ```

4. Configure environment variables in the Vercel dashboard:
   - `VITE_API_URL`: Your API endpoint URL
   - `VITE_ENABLE_ANALYTICS`: Set to `true` for production

### Deploying to Netlify

1. Install Netlify CLI (optional but recommended):
   ```bash
   npm install -g netlify-cli
   ```

2. Login to Netlify:
   ```bash
   netlify login
   ```

3. Deploy the application:
   ```bash
   netlify deploy --prod
   ```

4. Configure environment variables in the Netlify dashboard:
   - `VITE_API_URL`: Your API endpoint URL
   - `VITE_ENABLE_ANALYTICS`: Set to `true` for production

### Automated Deployment with GitHub Actions

See the `.github/workflows/cd-dev.yml` file for the automated deployment configuration that:

1. Builds the frontend with the appropriate environment variables
2. Deploys to Vercel/Netlify automatically on push to main/dev branches

## CDN Configuration

Both Vercel and Netlify provide CDN capabilities out of the box. Additional configurations:

1. **Cache Control Headers**:
   - HTML: `Cache-Control: public, max-age=0, must-revalidate`
   - Assets (JS/CSS/Images): `Cache-Control: public, max-age=31536000, immutable`

2. **Content Security Policy**:
   - Restrict external script sources
   - Limit connection to API domain only
   - Prevent inline scripts (when possible)

## Performance Monitoring

Free tier performance monitoring is available through:

1. **Vercel Analytics**: Basic page load and visitor metrics
2. **Netlify Analytics**: Request counts and popular pages
3. **Google PageSpeed Insights**: Regular manual checks

## Troubleshooting

Common issues and solutions:

1. **API Connection Issues**:
   - Verify the API endpoint URL is correct
   - Check CORS configuration on the backend
   - Ensure environment variables are properly set

2. **Build Failures**:
   - Check dependency versions compatibility
   - Verify Node.js version requirements
   - Review build logs for specific errors

3. **Performance Issues**:
   - Run Lighthouse audits to identify bottlenecks
   - Consider code splitting for large bundles
   - Optimize image sizes and formats

## Local Development

For local development with hot reloading:

```bash
npm run dev
```

This will start a development server with:
- Hot module replacement
- Source maps
- Development API proxy (configured in `vite.config.ts`)

## Updating the Frontend

When making updates to the frontend:

1. Make changes locally and test
2. Push to the GitHub repository
3. CI/CD pipeline will automatically deploy updates
4. Verify the deployment in production

## Technical Considerations

### Browser Compatibility

The frontend is built to support:
- Chrome (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Edge (latest 2 versions)

### Accessibility

The frontend implements:
- Semantic HTML elements
- ARIA attributes where necessary
- Keyboard navigation support
- Color contrast compliance

### Mobile Responsiveness

The application is fully responsive with:
- Mobile-first design approach
- Adaptive layouts for different screen sizes
- Touch-friendly UI elements 