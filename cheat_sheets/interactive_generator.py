#!/usr/bin/env python3
"""
Interactive Cheat Sheet Generator

A tool to generate customized cheat sheets based on specific algorithms, 
topics, or use cases for machine learning practitioners.

Usage:
    python interactive_generator.py
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import datetime

class CheatSheetGenerator:
    """Interactive cheat sheet generator for ML topics"""
    
    def __init__(self):
        """Initialize the generator with available topics and templates"""
        self.base_path = Path(__file__).parent
        self.output_path = self.base_path / "generated"
        self.output_path.mkdir(exist_ok=True)
        
        # Available cheat sheet categories and topics
        self.categories = {
            "algorithms": {
                "Linear Regression": "linear_regression",
                "Logistic Regression": "logistic_regression", 
                "Decision Trees": "decision_trees",
                "Random Forest": "random_forest",
                "SVM": "svm",
                "K-Means": "kmeans",
                "Neural Networks": "neural_networks",
                "Naive Bayes": "naive_bayes",
                "K-Nearest Neighbors": "knn",
                "Gradient Boosting": "gradient_boosting"
            },
            "preprocessing": {
                "Data Cleaning": "data_cleaning",
                "Feature Engineering": "feature_engineering",
                "Feature Selection": "feature_selection",
                "Scaling & Normalization": "scaling",
                "Handling Missing Data": "missing_data",
                "Categorical Encoding": "categorical_encoding"
            },
            "evaluation": {
                "Classification Metrics": "classification_metrics",
                "Regression Metrics": "regression_metrics",
                "Cross Validation": "cross_validation",
                "Model Selection": "model_selection",
                "Hyperparameter Tuning": "hyperparameter_tuning"
            },
            "visualization": {
                "Matplotlib": "matplotlib",
                "Seaborn": "seaborn",
                "Plotly": "plotly",
                "Data Exploration Plots": "exploration_plots"
            },
            "tools": {
                "Scikit-learn": "sklearn",
                "Pandas": "pandas",
                "NumPy": "numpy",
                "Jupyter Notebooks": "jupyter"
            },
            "math": {
                "Linear Algebra": "linear_algebra",
                "Statistics": "statistics",
                "Probability": "probability",
                "Calculus": "calculus"
            }
        }
        
        # Templates for different sections
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load content templates for different topics"""
        templates = {
            # Algorithm templates
            "linear_regression": {
                "title": "Linear Regression Cheat Sheet",
                "description": "Quick reference for linear regression implementation and theory",
                "sections": [
                    {
                        "name": "Key Concepts",
                        "content": [
                            "**Linear Regression Equation**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε",
                            "**Assumptions**: Linearity, Independence, Homoscedasticity, Normality",
                            "**Cost Function**: MSE = (1/n) Σ(yᵢ - ŷᵢ)²",
                            "**Normal Equation**: β = (XᵀX)⁻¹Xᵀy"
                        ]
                    },
                    {
                        "name": "Implementation",
                        "content": [
                            "```python",
                            "from sklearn.linear_model import LinearRegression",
                            "from sklearn.metrics import mean_squared_error, r2_score",
                            "",
                            "# Create and train model",
                            "model = LinearRegression()",
                            "model.fit(X_train, y_train)",
                            "",
                            "# Make predictions",
                            "y_pred = model.predict(X_test)",
                            "",
                            "# Evaluate",
                            "mse = mean_squared_error(y_test, y_pred)",
                            "r2 = r2_score(y_test, y_pred)",
                            "```"
                        ]
                    },
                    {
                        "name": "Parameters & Attributes",
                        "content": [
                            "**Parameters:**",
                            "- `fit_intercept`: Whether to calculate intercept (default: True)",
                            "- `normalize`: Whether to normalize features (deprecated)",
                            "",
                            "**Attributes:**",
                            "- `coef_`: Regression coefficients",
                            "- `intercept_`: Intercept term",
                            "- `score()`: R² coefficient of determination"
                        ]
                    }
                ]
            },
            
            "data_cleaning": {
                "title": "Data Cleaning Cheat Sheet", 
                "description": "Essential techniques for cleaning and preparing data",
                "sections": [
                    {
                        "name": "Missing Data",
                        "content": [
                            "```python",
                            "# Check for missing values",
                            "df.isnull().sum()",
                            "df.info()",
                            "",
                            "# Remove missing values",
                            "df.dropna()  # Drop rows with any NaN",
                            "df.dropna(axis=1)  # Drop columns with any NaN",
                            "df.dropna(subset=['col1', 'col2'])  # Drop if specific columns have NaN",
                            "",
                            "# Fill missing values",
                            "df.fillna(0)  # Fill with constant",
                            "df.fillna(df.mean())  # Fill with mean",
                            "df.fillna(method='forward')  # Forward fill",
                            "df.fillna(method='backward')  # Backward fill",
                            "```"
                        ]
                    },
                    {
                        "name": "Data Types & Conversion",
                        "content": [
                            "```python",
                            "# Check data types",
                            "df.dtypes",
                            "",
                            "# Convert data types",
                            "df['column'] = df['column'].astype('float64')",
                            "df['date'] = pd.to_datetime(df['date'])",
                            "df['category'] = df['category'].astype('category')",
                            "",
                            "# Automatic type inference",
                            "df = pd.read_csv('file.csv', dtype={'column': 'category'})",
                            "```"
                        ]
                    }
                ]
            },
            
            "classification_metrics": {
                "title": "Classification Metrics Cheat Sheet",
                "description": "Comprehensive guide to classification evaluation metrics", 
                "sections": [
                    {
                        "name": "Confusion Matrix",
                        "content": [
                            "```python",
                            "from sklearn.metrics import confusion_matrix, classification_report",
                            "",
                            "# Confusion matrix",
                            "cm = confusion_matrix(y_true, y_pred)",
                            "print(classification_report(y_true, y_pred))",
                            "",
                            "# Visualization",
                            "import seaborn as sns",
                            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')",
                            "```"
                        ]
                    },
                    {
                        "name": "Key Metrics",
                        "content": [
                            "**Accuracy**: (TP + TN) / (TP + TN + FP + FN)",
                            "**Precision**: TP / (TP + FP) - How many predicted positives are actually positive",
                            "**Recall (Sensitivity)**: TP / (TP + FN) - How many actual positives were identified",
                            "**Specificity**: TN / (TN + FP) - How many actual negatives were identified",
                            "**F1-Score**: 2 * (Precision × Recall) / (Precision + Recall)",
                            "",
                            "```python",
                            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score",
                            "",
                            "accuracy = accuracy_score(y_true, y_pred)",
                            "precision = precision_score(y_true, y_pred, average='weighted')",
                            "recall = recall_score(y_true, y_pred, average='weighted')",
                            "f1 = f1_score(y_true, y_pred, average='weighted')",
                            "```"
                        ]
                    }
                ]
            }
        }
        return templates
    
    def display_menu(self):
        """Display the main menu for category selection"""
        print("\n" + "="*60)
        print("🚀 ML CHEAT SHEET GENERATOR")
        print("="*60)
        print("\nAvailable Categories:")
        
        for i, (category, topics) in enumerate(self.categories.items(), 1):
            print(f"{i}. {category.title()} ({len(topics)} topics)")
        
        print(f"{len(self.categories) + 1}. Generate Custom Cheat Sheet")
        print(f"{len(self.categories) + 2}. View Generated Cheat Sheets")
        print("0. Exit")
        print("-"*60)
    
    def display_category_topics(self, category: str):
        """Display topics within a selected category"""
        topics = self.categories[category]
        print(f"\n📚 {category.title()} Topics:")
        print("-"*40)
        
        for i, topic in enumerate(topics.keys(), 1):
            print(f"{i}. {topic}")
        
        print("0. Back to main menu")
        print("-"*40)
    
    def generate_cheat_sheet(self, topic_key: str, custom_title: Optional[str] = None) -> str:
        """Generate a cheat sheet for the specified topic"""
        if topic_key in self.templates:
            template = self.templates[topic_key]
            title = custom_title or template["title"]
            
            content = [
                f"# {title}",
                f"\n> {template['description']}",
                f"\n**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "\n---\n"
            ]
            
            # Add table of contents
            content.append("## Table of Contents\n")
            for section in template["sections"]:
                content.append(f"- [{section['name']}](#{section['name'].lower().replace(' ', '-')})")
            content.append("\n---\n")
            
            # Add sections
            for section in template["sections"]:
                content.append(f"## {section['name']}\n")
                for line in section["content"]:
                    content.append(line)
                content.append("\n")
            
            # Add footer
            content.extend([
                "---\n",
                "## Additional Resources\n",
                "- [Scikit-learn Documentation](https://scikit-learn.org/stable/)",
                "- [Pandas Documentation](https://pandas.pydata.org/docs/)",
                "- [ML Algorithms Learning Repository](../README.md)",
                "\n---\n",
                f"*Generated by ML Cheat Sheet Generator - {datetime.datetime.now().year}*"
            ])
            
            return "\n".join(content)
        
        else:
            # Generate basic template for unknown topics
            content = [
                f"# {custom_title or topic_key.replace('_', ' ').title()} Cheat Sheet",
                f"\n**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "\n---\n",
                "## Overview\n",
                f"This cheat sheet covers {topic_key.replace('_', ' ')} in machine learning.\n",
                "## Key Concepts\n",
                "- Concept 1",
                "- Concept 2",
                "- Concept 3\n",
                "## Implementation\n",
                "```python",
                "# Sample code will go here",
                "# Add your implementation details",
                "```\n",
                "## Best Practices\n",
                "1. Practice 1",
                "2. Practice 2",
                "3. Practice 3\n",
                "---\n",
                f"*Generated by ML Cheat Sheet Generator - {datetime.datetime.now().year}*"
            ]
            return "\n".join(content)
    
    def generate_custom_cheat_sheet(self):
        """Generate a custom cheat sheet based on user selections"""
        print("\n🎯 CUSTOM CHEAT SHEET GENERATOR")
        print("-"*50)
        
        # Get custom title
        title = input("Enter cheat sheet title: ").strip()
        if not title:
            title = "Custom ML Cheat Sheet"
        
        print("\nSelect topics to include (enter numbers separated by commas):")
        all_topics = []
        topic_mapping = {}
        
        counter = 1
        for category, topics in self.categories.items():
            print(f"\n{category.upper()}:")
            for topic_name, topic_key in topics.items():
                print(f"  {counter}. {topic_name}")
                topic_mapping[counter] = (topic_key, topic_name)
                counter += 1
        
        print("\nExample: 1,3,5 (or press Enter to include all topics)")
        selection = input("Your selection: ").strip()
        
        if selection:
            try:
                selected_numbers = [int(x.strip()) for x in selection.split(",")]
                selected_topics = [topic_mapping[num] for num in selected_numbers if num in topic_mapping]
            except ValueError:
                print("Invalid selection. Including all topics...")
                selected_topics = list(topic_mapping.values())
        else:
            selected_topics = list(topic_mapping.values())
        
        # Generate combined cheat sheet
        content = [
            f"# {title}",
            f"\n**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Topics Included:** {len(selected_topics)} topics",
            "\n---\n"
        ]
        
        # Add table of contents
        content.append("## Table of Contents\n")
        for i, (_, topic_name) in enumerate(selected_topics, 1):
            content.append(f"{i}. [{topic_name}](#{topic_name.lower().replace(' ', '-').replace('&', '').replace('-', '-')})")
        content.append("\n---\n")
        
        # Add each selected topic
        for i, (topic_key, topic_name) in enumerate(selected_topics, 1):
            content.append(f"## {i}. {topic_name}\n")
            
            if topic_key in self.templates:
                template = self.templates[topic_key]
                for section in template["sections"]:
                    content.append(f"### {section['name']}\n")
                    for line in section["content"]:
                        content.append(line)
                    content.append("\n")
            else:
                content.extend([
                    f"### Key Concepts for {topic_name}\n",
                    "- Add key concepts here",
                    "- Implementation details",
                    "- Best practices\n",
                    "### Implementation\n",
                    "```python",
                    "# Implementation code for " + topic_name,
                    "```\n"
                ])
            
            content.append("---\n")
        
        # Add footer
        content.extend([
            "## Additional Resources\n",
            "- [Scikit-learn Documentation](https://scikit-learn.org/stable/)",
            "- [Pandas Documentation](https://pandas.pydata.org/docs/)",
            "- [ML Algorithms Learning Repository](../README.md)",
            "\n---\n",
            f"*Generated by ML Cheat Sheet Generator - {datetime.datetime.now().year}*"
        ])
        
        return "\n".join(content)
    
    def save_cheat_sheet(self, content: str, filename: str):
        """Save the generated cheat sheet to a file"""
        filepath = self.output_path / f"{filename}.md"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Cheat sheet saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ Error saving cheat sheet: {e}")
            return None
    
    def view_generated_sheets(self):
        """Display list of previously generated cheat sheets"""
        generated_files = list(self.output_path.glob("*.md"))
        
        if not generated_files:
            print("\n📭 No generated cheat sheets found.")
            return
        
        print(f"\n📋 Generated Cheat Sheets ({len(generated_files)} files):")
        print("-"*50)
        
        for i, filepath in enumerate(generated_files, 1):
            # Get file stats
            stat = filepath.stat()
            size = stat.st_size
            modified = datetime.datetime.fromtimestamp(stat.st_mtime)
            
            print(f"{i}. {filepath.name}")
            print(f"   Size: {size:,} bytes | Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
        
        print("\nOptions:")
        print("1. Open a cheat sheet")
        print("2. Delete a cheat sheet") 
        print("0. Back to main menu")
        
        choice = input("\nChoose option (0-2): ").strip()
        
        if choice == "1":
            self._open_cheat_sheet(generated_files)
        elif choice == "2":
            self._delete_cheat_sheet(generated_files)
    
    def _open_cheat_sheet(self, files: List[Path]):
        """Open a generated cheat sheet"""
        try:
            file_num = int(input(f"Enter file number (1-{len(files)}): "))
            if 1 <= file_num <= len(files):
                filepath = files[file_num - 1]
                
                # Try to open with system default
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    os.startfile(filepath)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", filepath])
                else:  # Linux
                    subprocess.run(["xdg-open", filepath])
                
                print(f"✅ Opened: {filepath.name}")
            else:
                print("❌ Invalid file number")
        except (ValueError, Exception) as e:
            print(f"❌ Error opening file: {e}")
    
    def _delete_cheat_sheet(self, files: List[Path]):
        """Delete a generated cheat sheet"""
        try:
            file_num = int(input(f"Enter file number to delete (1-{len(files)}): "))
            if 1 <= file_num <= len(files):
                filepath = files[file_num - 1]
                confirm = input(f"⚠️  Delete '{filepath.name}'? (y/N): ").strip().lower()
                
                if confirm == 'y':
                    filepath.unlink()
                    print(f"✅ Deleted: {filepath.name}")
                else:
                    print("❌ Deletion cancelled")
            else:
                print("❌ Invalid file number")
        except (ValueError, Exception) as e:
            print(f"❌ Error deleting file: {e}")
    
    def run(self):
        """Main application loop"""
        while True:
            self.display_menu()
            
            try:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "0":
                    print("👋 Thanks for using ML Cheat Sheet Generator!")
                    break
                
                elif choice == str(len(self.categories) + 1):
                    # Generate custom cheat sheet
                    content = self.generate_custom_cheat_sheet()
                    filename = input("\nEnter filename (without .md extension): ").strip()
                    if not filename:
                        filename = f"custom_cheat_sheet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    self.save_cheat_sheet(content, filename)
                
                elif choice == str(len(self.categories) + 2):
                    # View generated cheat sheets
                    self.view_generated_sheets()
                
                elif choice.isdigit() and 1 <= int(choice) <= len(self.categories):
                    # Select category
                    category_names = list(self.categories.keys())
                    selected_category = category_names[int(choice) - 1]
                    
                    while True:
                        self.display_category_topics(selected_category)
                        
                        topic_choice = input("\nEnter topic number: ").strip()
                        
                        if topic_choice == "0":
                            break
                        
                        elif topic_choice.isdigit():
                            topic_names = list(self.categories[selected_category].keys())
                            topic_index = int(topic_choice) - 1
                            
                            if 0 <= topic_index < len(topic_names):
                                topic_name = topic_names[topic_index]
                                topic_key = self.categories[selected_category][topic_name]
                                
                                # Generate cheat sheet
                                content = self.generate_cheat_sheet(topic_key)
                                
                                # Save with default filename
                                filename = f"{topic_key}_cheat_sheet_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                self.save_cheat_sheet(content, filename)
                                
                                input("\nPress Enter to continue...")
                            else:
                                print("❌ Invalid topic number")
                        else:
                            print("❌ Invalid input")
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

def main():
    """Main entry point"""
    try:
        generator = CheatSheetGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()