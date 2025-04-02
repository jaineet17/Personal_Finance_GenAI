#!/bin/bash

# Create clean directory
CLEAN_DIR="../finance-llm-clean"
rm -rf $CLEAN_DIR
mkdir -p $CLEAN_DIR

# Copy source code and essential files
echo "Copying source files..."
cp -r src $CLEAN_DIR/
cp -r docs $CLEAN_DIR/
cp -r tests $CLEAN_DIR/
cp -r .github $CLEAN_DIR/
cp -r frontend $CLEAN_DIR/

# Copy important configuration files
echo "Copying configuration files..."
cp .gitignore $CLEAN_DIR/
cp .env.sample $CLEAN_DIR/
cp README.md $CLEAN_DIR/
cp Dockerfile $CLEAN_DIR/
cp deploy_fargate.sh $CLEAN_DIR/
cp requirements.txt $CLEAN_DIR/
cp requirements-lambda.txt $CLEAN_DIR/
cp serverless.yml $CLEAN_DIR/
cp ssm-parameter-policy.json $CLEAN_DIR/

# Create empty directories to maintain structure
echo "Creating directory structure..."
mkdir -p $CLEAN_DIR/data/raw
mkdir -p $CLEAN_DIR/data/processed
mkdir -p $CLEAN_DIR/data/vectors
mkdir -p $CLEAN_DIR/test_output

# Create .gitkeep files
touch $CLEAN_DIR/data/raw/.gitkeep
touch $CLEAN_DIR/data/processed/.gitkeep
touch $CLEAN_DIR/data/vectors/.gitkeep
touch $CLEAN_DIR/test_output/.gitkeep

# Remove any large files or binaries that might have been copied
echo "Cleaning up any large files..."
find $CLEAN_DIR -type f -size +10M -delete
find $CLEAN_DIR -name "*.pyc" -delete
find $CLEAN_DIR -name "__pycache__" -exec rm -rf {} +
find $CLEAN_DIR -name "node_modules" -exec rm -rf {} +
find $CLEAN_DIR -name ".DS_Store" -delete

echo "Clean repository prepared at $CLEAN_DIR"
echo "Size of clean repository:"
du -sh $CLEAN_DIR

echo "You can now:"
echo "1. cd $CLEAN_DIR"
echo "2. git init"
echo "3. git add ."
echo "4. git commit -m 'Initial commit'"
echo "5. git remote add origin https://github.com/jaineet17/Personal_Finance.git"
echo "6. git push -u origin main" 