#!/usr/bin/env python3
"""
Algorithm Recommendation Engine
==============================

Interactive script to recommend the best ML algorithms based on:
- Problem type (regression, classification, clustering)
- Dataset characteristics (size, features, etc.)
- User requirements (interpretability, speed, accuracy)
- Experience level

Usage: uv run python algorithm_recommender.py
"""

import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AlgorithmRecommendation:
    name: str
    type: str
    confidence: float
    reason: str
    implementation_path: str
    scenarios: List[str]

class AlgorithmRecommender:
    """Intelligent algorithm recommendation system"""
    
    def __init__(self):
        self.algorithms = {
            # Regression Algorithms
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'regression',
                'path': './01_supervised_learning/01_linear_regression/',
                'scenarios': ['house_prices', 'temperature', 'salary', 'stocks'],
                'best_for': ['linear_relationships', 'interpretability', 'small_datasets'],
                'min_samples': 30,
                'max_samples': float('inf'),
                'speed': 5,
                'accuracy_linear': 5,
                'accuracy_nonlinear': 2,
                'interpretability': 5,
                'complexity': 1
            },
            'xgboost_regression': {
                'name': 'XGBoost (Regression)',
                'type': 'regression', 
                'path': './04_advanced_topics/04_xgboost/',
                'scenarios': ['sales_forecast', 'credit_risk'],
                'best_for': ['complex_patterns', 'high_accuracy', 'large_datasets'],
                'min_samples': 100,
                'max_samples': 10000000,
                'speed': 2,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 5,
                'interpretability': 2,
                'complexity': 4
            },
            'random_forest_regression': {
                'name': 'Random Forest (Regression)',
                'type': 'regression',
                'path': './01_supervised_learning/04_random_forest/',
                'scenarios': ['general_regression'],
                'best_for': ['balanced_performance', 'feature_importance'],
                'min_samples': 100,
                'max_samples': 1000000,
                'speed': 3,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 4,
                'interpretability': 3,
                'complexity': 2
            },
            
            # Classification Algorithms  
            'logistic_regression': {
                'name': 'Logistic Regression',
                'type': 'classification',
                'path': './01_supervised_learning/02_logistic_regression/',
                'scenarios': ['binary_classification'],
                'best_for': ['binary_problems', 'interpretability', 'baseline'],
                'min_samples': 50,
                'max_samples': float('inf'),
                'speed': 5,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 3,
                'interpretability': 5,
                'complexity': 1
            },
            'xgboost_classification': {
                'name': 'XGBoost (Classification)', 
                'type': 'classification',
                'path': './04_advanced_topics/04_xgboost/',
                'scenarios': ['kaggle', 'credit_risk', 'churn', 'medical'],
                'best_for': ['high_accuracy', 'complex_patterns', 'competitions'],
                'min_samples': 100,
                'max_samples': 10000000,
                'speed': 2,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 5,
                'interpretability': 2,
                'complexity': 4
            },
            'random_forest_classification': {
                'name': 'Random Forest (Classification)',
                'type': 'classification',
                'path': './01_supervised_learning/04_random_forest/',
                'scenarios': ['general_classification'],
                'best_for': ['balanced_performance', 'robust', 'feature_importance'],
                'min_samples': 100,
                'max_samples': 1000000,
                'speed': 3,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 4,
                'interpretability': 3,
                'complexity': 2
            },
            'svm': {
                'name': 'Support Vector Machine',
                'type': 'classification',
                'path': './01_supervised_learning/05_svm/',
                'scenarios': ['high_dimensional', 'text_classification'],
                'best_for': ['high_dimensional', 'kernel_tricks', 'robust'],
                'min_samples': 100,
                'max_samples': 50000,
                'speed': 2,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 4,
                'interpretability': 2,
                'complexity': 3
            },
            'naive_bayes': {
                'name': 'Naive Bayes',
                'type': 'classification',
                'path': './01_supervised_learning/06_naive_bayes/',
                'scenarios': ['text_classification', 'small_datasets'],
                'best_for': ['text_data', 'small_datasets', 'fast_training'],
                'min_samples': 50,
                'max_samples': float('inf'),
                'speed': 5,
                'accuracy_linear': 3,
                'accuracy_nonlinear': 3,
                'interpretability': 4,
                'complexity': 1
            },
            
            # Clustering Algorithms
            'kmeans': {
                'name': 'K-Means Clustering',
                'type': 'clustering',
                'path': './02_unsupervised_learning/01_kmeans/',
                'scenarios': ['customer_segmentation'],
                'best_for': ['spherical_clusters', 'known_k', 'fast'],
                'min_samples': 100,
                'max_samples': 1000000,
                'speed': 4,
                'accuracy_linear': 4,
                'accuracy_nonlinear': 3,
                'interpretability': 4,
                'complexity': 2
            }
        }
    
    def interactive_recommendation(self) -> List[AlgorithmRecommendation]:
        """Interactive questionnaire to recommend algorithms"""
        print("🔬 ML Algorithm Recommendation Engine")
        print("=====================================\n")
        
        # Collect user requirements
        answers = self._collect_user_input()
        
        # Calculate recommendations
        recommendations = self._calculate_recommendations(answers)
        
        # Display results
        self._display_recommendations(recommendations, answers)
        
        return recommendations
    
    def _collect_user_input(self) -> Dict:
        """Collect user requirements through interactive questions"""
        answers = {}
        
        # Problem type
        print("1. What type of problem are you solving?")
        print("   a) Predict a continuous number (regression)")
        print("   b) Predict a category/class (classification)")  
        print("   c) Find groups/patterns in data (clustering)")
        problem_type = input("\nEnter your choice (a/b/c): ").lower().strip()
        
        problem_map = {'a': 'regression', 'b': 'classification', 'c': 'clustering'}
        answers['problem_type'] = problem_map.get(problem_type, 'regression')
        
        # Dataset size
        print("\n2. How many samples (rows) do you have?")
        print("   a) Small (< 1,000)")
        print("   b) Medium (1,000 - 100,000)")
        print("   c) Large (> 100,000)")
        size = input("\nEnter your choice (a/b/c): ").lower().strip()
        
        size_map = {'a': 500, 'b': 50000, 'c': 500000}
        answers['dataset_size'] = size_map.get(size, 50000)
        
        # Data relationship
        if answers['problem_type'] in ['regression', 'classification']:
            print("\n3. What kind of relationship do you expect in your data?")
            print("   a) Simple/Linear patterns")
            print("   b) Complex/Non-linear patterns") 
            print("   c) Not sure")
            relationship = input("\nEnter your choice (a/b/c): ").lower().strip()
            answers['data_complexity'] = 'linear' if relationship == 'a' else 'nonlinear'
        else:
            answers['data_complexity'] = 'nonlinear'
            
        # Interpretability importance
        print("\n4. How important is it to understand HOW the model makes decisions?")
        print("   a) Very important (need to explain to others)")
        print("   b) Somewhat important")
        print("   c) Not important (just need accuracy)")
        interpret = input("\nEnter your choice (a/b/c): ").lower().strip()
        
        interpret_map = {'a': 5, 'b': 3, 'c': 1}
        answers['interpretability_need'] = interpret_map.get(interpret, 3)
        
        # Experience level
        print("\n5. What's your machine learning experience level?")
        print("   a) Beginner (new to ML)")
        print("   b) Intermediate (some experience)")
        print("   c) Advanced (experienced)")
        exp = input("\nEnter your choice (a/b/c): ").lower().strip()
        
        exp_map = {'a': 1, 'b': 2, 'c': 3}
        answers['experience'] = exp_map.get(exp, 2)
        
        # Time for tuning
        print("\n6. How much time do you have for model optimization?")
        print("   a) Minimal (want quick results)")
        print("   b) Some time (moderate tuning)")
        print("   c) Lots of time (extensive optimization)")
        time = input("\nEnter your choice (a/b/c): ").lower().strip()
        
        time_map = {'a': 1, 'b': 2, 'c': 3}
        answers['tuning_time'] = time_map.get(time, 2)
        
        return answers
    
    def _calculate_recommendations(self, answers: Dict) -> List[AlgorithmRecommendation]:
        """Calculate algorithm recommendations based on user answers"""
        recommendations = []
        
        # Filter algorithms by problem type
        relevant_algorithms = {
            k: v for k, v in self.algorithms.items() 
            if v['type'] == answers['problem_type']
        }
        
        for algo_key, algo_info in relevant_algorithms.items():
            confidence = self._calculate_confidence_score(algo_info, answers)
            reason = self._generate_reason(algo_info, answers)
            
            recommendation = AlgorithmRecommendation(
                name=algo_info['name'],
                type=algo_info['type'],
                confidence=confidence,
                reason=reason,
                implementation_path=algo_info['path'],
                scenarios=algo_info['scenarios']
            )
            recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:3]  # Top 3 recommendations
    
    def _calculate_confidence_score(self, algo_info: Dict, answers: Dict) -> float:
        """Calculate confidence score for an algorithm based on user requirements"""
        score = 0.0
        max_score = 0.0
        
        # Dataset size compatibility (20% weight)
        if answers['dataset_size'] >= algo_info['min_samples'] and \
           answers['dataset_size'] <= algo_info['max_samples']:
            score += 20
        max_score += 20
        
        # Speed requirement (15% weight) - higher for beginners and less tuning time
        speed_importance = (4 - answers['experience']) + (4 - answers['tuning_time'])
        speed_score = (algo_info['speed'] / 5) * speed_importance * 3
        score += speed_score
        max_score += 15
        
        # Accuracy requirement (25% weight)
        if answers['data_complexity'] == 'linear':
            accuracy_score = (algo_info['accuracy_linear'] / 5) * 25
        else:
            accuracy_score = (algo_info['accuracy_nonlinear'] / 5) * 25
        score += accuracy_score
        max_score += 25
        
        # Interpretability (20% weight)
        interpret_score = (algo_info['interpretability'] / 5) * (answers['interpretability_need'] / 5) * 20
        score += interpret_score
        max_score += 20
        
        # Complexity/Experience match (20% weight)
        complexity_match = 5 - abs(algo_info['complexity'] - answers['experience'])
        complexity_score = (complexity_match / 5) * 20
        score += complexity_score
        max_score += 20
        
        return min(100, (score / max_score) * 100)
    
    def _generate_reason(self, algo_info: Dict, answers: Dict) -> str:
        """Generate explanation for why this algorithm was recommended"""
        reasons = []
        
        if answers['data_complexity'] == 'linear' and algo_info['accuracy_linear'] >= 4:
            reasons.append("excellent for linear relationships")
        elif answers['data_complexity'] == 'nonlinear' and algo_info['accuracy_nonlinear'] >= 4:
            reasons.append("handles complex non-linear patterns well")
            
        if answers['interpretability_need'] >= 4 and algo_info['interpretability'] >= 4:
            reasons.append("highly interpretable")
        elif answers['interpretability_need'] <= 2 and algo_info['accuracy_nonlinear'] >= 4:
            reasons.append("prioritizes accuracy over interpretability")
            
        if answers['dataset_size'] <= 1000 and algo_info['min_samples'] <= 100:
            reasons.append("works well with small datasets")
        elif answers['dataset_size'] >= 100000 and algo_info['max_samples'] >= 100000:
            reasons.append("scales to large datasets")
            
        if answers['experience'] == 1 and algo_info['complexity'] <= 2:
            reasons.append("beginner-friendly")
        elif answers['experience'] == 3 and algo_info['complexity'] >= 3:
            reasons.append("suitable for advanced users")
            
        return "Good choice because it's " + ", ".join(reasons) if reasons else "balanced performance across your requirements"
    
    def _display_recommendations(self, recommendations: List[AlgorithmRecommendation], answers: Dict):
        """Display recommendations to user"""
        print("\n" + "="*60)
        print("🎯 ALGORITHM RECOMMENDATIONS")
        print("="*60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.name}")
            print(f"   Confidence: {rec.confidence:.1f}%")
            print(f"   Reason: {rec.reason}")
            print(f"   Learn more: {rec.implementation_path}")
            
            if rec.scenarios:
                print(f"   Try scenarios: {', '.join(rec.scenarios)}")
        
        print("\n" + "="*60)
        print("📚 NEXT STEPS:")
        print("1. Click on the path links above to learn about your recommended algorithms")
        print("2. Read the theory and concepts in the README files")
        print("3. Try the practical scenarios with real data")
        print("4. Compare performance using different algorithms")
        print("\n💡 TIP: Start with the highest confidence recommendation!")
        
        # Show command to run scenarios
        if recommendations:
            top_rec = recommendations[0]
            if top_rec.scenarios:
                scenario = top_rec.scenarios[0]
                print(f"\n🚀 Quick start command:")
                if 'xgboost' in top_rec.implementation_path:
                    print(f"uv run python sklearn_example.py --scenario {scenario}")
                else:
                    print(f"uv run python implementation.py --scenario {scenario}")

def main():
    """Main function"""
    try:
        recommender = AlgorithmRecommender()
        recommendations = recommender.interactive_recommendation()
        
        print("\n" + "="*60)
        print("📊 Need more detailed comparisons?")
        print("Check out: ALGORITHM_COMPARISON.md")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nRecommendation cancelled. Run the script again anytime!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()