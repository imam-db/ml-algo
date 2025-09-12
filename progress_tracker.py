#!/usr/bin/env python3
"""
ML Learning Progress Tracker
============================

Interactive script to track your machine learning learning progress:
- Update algorithm completion status
- Track project milestones
- Award badges and achievements
- Generate progress reports
- Set learning goals

Usage: uv run python progress_tracker.py
"""

import json
import os
import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Achievement:
    id: str
    name: str
    description: str
    category: str
    tier: str  # bronze, silver, gold, legendary
    earned: bool = False
    date_earned: str = ""

@dataclass
class AlgorithmProgress:
    name: str
    theory_complete: bool = False
    implementation_complete: bool = False
    scenarios_complete: bool = False
    projects_complete: bool = False
    badge_earned: bool = False
    completion_percentage: int = 0

@dataclass
class ProjectProgress:
    name: str
    description: str
    milestones: List[str]
    completed_milestones: List[bool]
    status: str = "not_started"  # not_started, in_progress, completed
    completion_percentage: int = 0

@dataclass
class LearningStats:
    total_study_days: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    last_study_date: str = ""
    algorithms_mastered: int = 0
    projects_completed: int = 0
    badges_earned: int = 0
    weekly_goals: List[str] = None
    focus_area: str = ""

class ProgressTracker:
    """ML Learning Progress Tracking System"""
    
    def __init__(self):
        self.progress_file = Path("ml_progress.json")
        self.algorithms = [
            "Linear Regression", "Logistic Regression", "Decision Trees", 
            "Random Forest", "SVM", "Naive Bayes", "K-NN", 
            "Neural Networks", "K-Means", "PCA", "XGBoost"
        ]
        
        self.projects = [
            ProjectProgress(
                name="My First ML Model",
                description="Build your first predictive model using Linear Regression",
                milestones=[
                    "Load and explore dataset",
                    "Handle missing values and outliers", 
                    "Train linear regression model",
                    "Evaluate model performance",
                    "Create predictions on new data",
                    "Write summary of findings"
                ],
                completed_milestones=[False] * 6
            ),
            ProjectProgress(
                name="Business Intelligence Dashboard", 
                description="Create customer insights using classification and clustering",
                milestones=[
                    "Perform exploratory data analysis",
                    "Predict customer churn (classification)",
                    "Segment customers into groups (clustering)", 
                    "Create business recommendations",
                    "Build visualization dashboard",
                    "Present findings to stakeholders"
                ],
                completed_milestones=[False] * 6
            ),
            ProjectProgress(
                name="ML Competition Challenge",
                description="Compete in Kaggle-style competition with ensemble methods",
                milestones=[
                    "Advanced feature engineering",
                    "Hyperparameter optimization", 
                    "Model stacking and ensembling",
                    "Cross-validation strategy",
                    "Final model deployment",
                    "Achieve top 25% performance"
                ],
                completed_milestones=[False] * 6
            ),
            ProjectProgress(
                name="End-to-End ML System",
                description="Build production-ready ML system with monitoring",
                milestones=[
                    "Design system architecture",
                    "Implement data pipeline",
                    "Model training and validation",
                    "API deployment", 
                    "Monitoring and logging",
                    "A/B testing framework"
                ],
                completed_milestones=[False] * 6
            )
        ]
        
        self.achievements = self._create_achievements()
        self.data = self._load_progress()
    
    def _create_achievements(self) -> List[Achievement]:
        """Create all available achievements"""
        achievements = []
        
        # Bronze badges (Beginner)
        bronze_achievements = [
            ("hello_ml", "Hello ML", "Complete your first scenario", "first_steps"),
            ("data_explorer", "Data Explorer", "Analyze your first dataset", "first_steps"),
            ("predictor", "Predictor", "Make your first prediction", "first_steps"), 
            ("evaluator", "Evaluator", "Calculate model performance metrics", "first_steps"),
            ("linear_learner", "Linear Learner", "Master Linear Regression", "algorithm_basics"),
            ("classifier", "Classifier", "Master Logistic Regression", "algorithm_basics"),
            ("clusterer", "Clusterer", "Master K-Means", "algorithm_basics"),
            ("tree_hugger", "Tree Hugger", "Master Decision Trees", "algorithm_basics")
        ]
        
        for aid, name, desc, cat in bronze_achievements:
            achievements.append(Achievement(aid, name, desc, cat, "bronze"))
        
        # Silver badges (Intermediate)
        silver_achievements = [
            ("from_scratch", "From Scratch", "Implement 3 algorithms without libraries", "technical_skills"),
            ("feature_engineer", "Feature Engineer", "Create effective feature engineering pipeline", "technical_skills"),
            ("hyperparameter_hunter", "Hyperparameter Hunter", "Master hyperparameter tuning", "technical_skills"),
            ("cross_validator", "Cross Validator", "Implement robust validation strategies", "technical_skills"),
            ("ensemble_master", "Ensemble Master", "Master Random Forest", "advanced_algorithms"),
            ("svm_specialist", "SVM Specialist", "Master Support Vector Machines", "advanced_algorithms"),
            ("boosting_expert", "Boosting Expert", "Master XGBoost", "advanced_algorithms"),
            ("neural_navigator", "Neural Navigator", "Master Neural Networks", "advanced_algorithms")
        ]
        
        for aid, name, desc, cat in silver_achievements:
            achievements.append(Achievement(aid, name, desc, cat, "silver"))
        
        # Gold badges (Advanced)
        gold_achievements = [
            ("algorithm_sage", "Algorithm Sage", "Master all 10+ algorithms", "mastery"),
            ("project_champion", "Project Champion", "Complete all 4 progressive projects", "mastery"),
            ("competition_crusher", "Competition Crusher", "Achieve top performance in competition", "mastery"),
            ("ml_mentor", "ML Mentor", "Help others learn (contribute to community)", "mastery"),
            ("original_researcher", "Original Researcher", "Implement paper from scratch", "innovation")
        ]
        
        for aid, name, desc, cat in gold_achievements:
            achievements.append(Achievement(aid, name, desc, cat, "gold"))
        
        # Legendary badges (Expert)
        legendary_achievements = [
            ("ml_grandmaster", "ML Grandmaster", "Complete entire curriculum + contribute", "elite_status"),
            ("research_pioneer", "Research Pioneer", "Publish original ML research", "elite_status"),
            ("industry_leader", "Industry Leader", "Lead ML team in production environment", "elite_status"),
            ("community_legend", "Community Legend", "Significant contributions to ML community", "elite_status")
        ]
        
        for aid, name, desc, cat in legendary_achievements:
            achievements.append(Achievement(aid, name, desc, cat, "legendary"))
        
        return achievements
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from JSON file or create new"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
        else:
            # Create initial progress structure
            data = {
                "algorithms": {algo: asdict(AlgorithmProgress(algo)) for algo in self.algorithms},
                "projects": {proj.name: asdict(proj) for proj in self.projects},
                "achievements": {ach.id: asdict(ach) for ach in self.achievements},
                "stats": asdict(LearningStats()),
                "created_date": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat()
            }
            self._save_progress(data)
        
        return data
    
    def _save_progress(self, data: Dict[str, Any]):
        """Save progress to JSON file"""
        data["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def main_menu(self):
        """Display main menu and handle user interaction"""
        while True:
            print("\n" + "="*60)
            print("🎯 ML LEARNING PROGRESS TRACKER")
            print("="*60)
            print("\n📊 Current Progress Summary:")
            self._show_progress_summary()
            
            print("\n🎯 What would you like to do?")
            print("1. 📈 Update Algorithm Progress")
            print("2. 🚀 Update Project Progress") 
            print("3. 🏆 View Achievements")
            print("4. 📊 Generate Progress Report")
            print("5. 🎯 Set Learning Goals")
            print("6. 📈 Update Study Streak")
            print("7. 🔍 View Detailed Statistics")
            print("8. 📤 Export Progress")
            print("9. ❌ Exit")
            
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                self._update_algorithm_progress()
            elif choice == '2':
                self._update_project_progress()
            elif choice == '3':
                self._view_achievements()
            elif choice == '4':
                self._generate_progress_report()
            elif choice == '5':
                self._set_learning_goals()
            elif choice == '6':
                self._update_study_streak()
            elif choice == '7':
                self._view_detailed_statistics()
            elif choice == '8':
                self._export_progress()
            elif choice == '9':
                print("\n🎉 Keep up the great learning! See you next time!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _show_progress_summary(self):
        """Show quick progress summary"""
        stats = self.data["stats"]
        algorithms = self.data["algorithms"]
        projects = self.data["projects"]
        achievements = self.data["achievements"]
        
        # Calculate completion percentages
        algo_completed = sum(1 for a in algorithms.values() if a["badge_earned"])
        total_algos = len(algorithms)
        
        proj_completed = sum(1 for p in projects.values() if p["status"] == "completed")
        total_projects = len(projects)
        
        badges_earned = sum(1 for a in achievements.values() if a["earned"])
        total_badges = len(achievements)
        
        print(f"🎯 Overall Progress: {self._calculate_overall_completion():.0f}%")
        print(f"📚 Algorithms Mastered: {algo_completed}/{total_algos}")
        print(f"🚀 Projects Completed: {proj_completed}/{total_projects}")
        print(f"🏆 Badges Earned: {badges_earned}/{total_badges}")
        print(f"🔥 Current Streak: {stats['current_streak']} days")
    
    def _calculate_overall_completion(self) -> float:
        """Calculate overall completion percentage"""
        total_score = 0
        max_score = 0
        
        # Algorithm progress (60% weight)
        for algo_data in self.data["algorithms"].values():
            total_score += algo_data["completion_percentage"] * 0.6 / len(self.algorithms)
            max_score += 60 / len(self.algorithms)
        
        # Project progress (30% weight)  
        for proj_data in self.data["projects"].values():
            total_score += proj_data["completion_percentage"] * 0.3 / len(self.projects)
            max_score += 30 / len(self.projects)
        
        # Achievement progress (10% weight)
        badges_earned = sum(1 for a in self.data["achievements"].values() if a["earned"])
        total_score += (badges_earned / len(self.achievements)) * 10
        max_score += 10
        
        return (total_score / max_score) * 100
    
    def _update_algorithm_progress(self):
        """Update progress for specific algorithm"""
        print("\n📚 Algorithm Progress Update")
        print("="*40)
        
        # Show current algorithm progress
        for i, algo in enumerate(self.algorithms, 1):
            progress = self.data["algorithms"][algo]["completion_percentage"]
            status = "✅ Mastered" if self.data["algorithms"][algo]["badge_earned"] else f"{progress}%"
            print(f"{i:2d}. {algo}: {status}")
        
        choice = input(f"\nSelect algorithm to update (1-{len(self.algorithms)}): ").strip()
        
        try:
            algo_idx = int(choice) - 1
            if 0 <= algo_idx < len(self.algorithms):
                algo_name = self.algorithms[algo_idx]
                self._update_specific_algorithm(algo_name)
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def _update_specific_algorithm(self, algo_name: str):
        """Update specific algorithm progress"""
        algo_data = self.data["algorithms"][algo_name]
        
        print(f"\n📈 Updating {algo_name} Progress")
        print("="*40)
        
        # Show current status
        print(f"Theory Understanding: {'✅' if algo_data['theory_complete'] else '❌'}")
        print(f"Implementation Skills: {'✅' if algo_data['implementation_complete'] else '❌'}")
        print(f"Scenario Completion: {'✅' if algo_data['scenarios_complete'] else '❌'}")
        print(f"Project Application: {'✅' if algo_data['projects_complete'] else '❌'}")
        print(f"Badge Earned: {'✅' if algo_data['badge_earned'] else '❌'}")
        
        # Update each component
        if not algo_data["theory_complete"]:
            if input("\n📖 Mark theory as complete? (y/n): ").lower() == 'y':
                algo_data["theory_complete"] = True
                print("✅ Theory marked complete!")
        
        if not algo_data["implementation_complete"]:
            if input("💻 Mark implementation as complete? (y/n): ").lower() == 'y':
                algo_data["implementation_complete"] = True
                print("✅ Implementation marked complete!")
        
        if not algo_data["scenarios_complete"]:
            if input("🎯 Mark all scenarios as complete? (y/n): ").lower() == 'y':
                algo_data["scenarios_complete"] = True
                print("✅ Scenarios marked complete!")
        
        if not algo_data["projects_complete"]:
            if input("🚀 Mark project application as complete? (y/n): ").lower() == 'y':
                algo_data["projects_complete"] = True
                print("✅ Projects marked complete!")
        
        # Calculate completion percentage
        completed = sum([
            algo_data["theory_complete"],
            algo_data["implementation_complete"], 
            algo_data["scenarios_complete"],
            algo_data["projects_complete"]
        ])
        algo_data["completion_percentage"] = int((completed / 4) * 100)
        
        # Check for badge eligibility
        if completed == 4 and not algo_data["badge_earned"]:
            algo_data["badge_earned"] = True
            print(f"\n🏆 CONGRATULATIONS! You've earned the {algo_name} Mastery Badge!")
            self._award_algorithm_badge(algo_name)
        
        # Update stats
        self.data["stats"]["algorithms_mastered"] = sum(
            1 for a in self.data["algorithms"].values() if a["badge_earned"]
        )
        
        self._save_progress(self.data)
        print(f"\n✅ Progress saved for {algo_name}!")
    
    def _award_algorithm_badge(self, algo_name: str):
        """Award specific algorithm badge"""
        badge_mapping = {
            "Linear Regression": "linear_learner",
            "Logistic Regression": "classifier", 
            "K-Means": "clusterer",
            "Decision Trees": "tree_hugger",
            "Random Forest": "ensemble_master",
            "SVM": "svm_specialist",
            "XGBoost": "boosting_expert",
            "Neural Networks": "neural_navigator"
        }
        
        badge_id = badge_mapping.get(algo_name)
        if badge_id and badge_id in self.data["achievements"]:
            self.data["achievements"][badge_id]["earned"] = True
            self.data["achievements"][badge_id]["date_earned"] = datetime.datetime.now().isoformat()
            
            # Check for special achievements
            self._check_special_achievements()
    
    def _check_special_achievements(self):
        """Check and award special achievements"""
        # Algorithm Sage - Master all algorithms
        algos_mastered = sum(1 for a in self.data["algorithms"].values() if a["badge_earned"])
        if algos_mastered >= len(self.algorithms):
            if not self.data["achievements"]["algorithm_sage"]["earned"]:
                self.data["achievements"]["algorithm_sage"]["earned"] = True
                self.data["achievements"]["algorithm_sage"]["date_earned"] = datetime.datetime.now().isoformat()
                print("\n🌟 LEGENDARY ACHIEVEMENT UNLOCKED: Algorithm Sage!")
        
        # From Scratch - Implement 3+ algorithms
        from_scratch_count = sum(1 for a in self.data["algorithms"].values() if a["implementation_complete"])
        if from_scratch_count >= 3:
            if not self.data["achievements"]["from_scratch"]["earned"]:
                self.data["achievements"]["from_scratch"]["earned"] = True
                self.data["achievements"]["from_scratch"]["date_earned"] = datetime.datetime.now().isoformat()
                print("\n🥈 SILVER BADGE UNLOCKED: From Scratch!")
    
    def _view_achievements(self):
        """Display all achievements and their status"""
        print("\n🏆 ACHIEVEMENT GALLERY")
        print("="*60)
        
        tiers = ["bronze", "silver", "gold", "legendary"]
        tier_icons = {"bronze": "🥉", "silver": "🥈", "gold": "🥇", "legendary": "💎"}
        
        for tier in tiers:
            tier_achievements = [a for a in self.data["achievements"].values() if a["tier"] == tier]
            earned_count = sum(1 for a in tier_achievements if a["earned"])
            
            print(f"\n{tier_icons[tier]} {tier.upper()} BADGES ({earned_count}/{len(tier_achievements)})")
            print("-" * 40)
            
            for ach in tier_achievements:
                status = "✅" if ach["earned"] else "⬜"
                date_str = f" (Earned: {ach['date_earned'][:10]})" if ach["earned"] else ""
                print(f"{status} {ach['name']}: {ach['description']}{date_str}")
    
    def _generate_progress_report(self):
        """Generate comprehensive progress report"""
        print("\n📊 COMPREHENSIVE PROGRESS REPORT")
        print("="*60)
        
        # Overall statistics
        overall = self._calculate_overall_completion()
        print(f"\n🎯 Overall Completion: {overall:.1f}%")
        
        # Algorithm progress
        print(f"\n📚 ALGORITHM MASTERY PROGRESS")
        print("-" * 40)
        for algo, data in self.data["algorithms"].items():
            progress_bar = "█" * (data["completion_percentage"] // 10) + "░" * (10 - data["completion_percentage"] // 10)
            print(f"{algo:20s}: {progress_bar} {data['completion_percentage']}%")
        
        # Project progress
        print(f"\n🚀 PROJECT COMPLETION STATUS")
        print("-" * 40)
        for proj_name, proj_data in self.data["projects"].items():
            status_icon = {"not_started": "⏳", "in_progress": "🔄", "completed": "✅"}
            icon = status_icon.get(proj_data["status"], "❓")
            print(f"{icon} {proj_name}: {proj_data['completion_percentage']}%")
        
        # Learning statistics
        stats = self.data["stats"]
        print(f"\n📈 LEARNING STATISTICS")
        print("-" * 40)
        print(f"Total Study Days: {stats['total_study_days']}")
        print(f"Current Streak: {stats['current_streak']} days 🔥")
        print(f"Longest Streak: {stats['longest_streak']} days 🏆")
        print(f"Last Study Date: {stats['last_study_date']}")
        
        # Badge summary
        badge_counts = {"bronze": 0, "silver": 0, "gold": 0, "legendary": 0}
        for ach in self.data["achievements"].values():
            if ach["earned"]:
                badge_counts[ach["tier"]] += 1
        
        print(f"\n🏆 BADGES EARNED")
        print("-" * 40)
        print(f"🥉 Bronze: {badge_counts['bronze']}")
        print(f"🥈 Silver: {badge_counts['silver']}")
        print(f"🥇 Gold: {badge_counts['gold']}")
        print(f"💎 Legendary: {badge_counts['legendary']}")
        
        input("\nPress Enter to continue...")
    
    def _set_learning_goals(self):
        """Set weekly learning goals and focus areas"""
        print("\n🎯 LEARNING GOAL SETTING")
        print("="*40)
        
        current_focus = self.data["stats"].get("focus_area", "")
        current_goals = self.data["stats"].get("weekly_goals", [])
        
        print(f"Current Focus Area: {current_focus or 'Not set'}")
        print(f"Current Weekly Goals: {len(current_goals)} goals set")
        
        if input("\nUpdate focus area? (y/n): ").lower() == 'y':
            new_focus = input("Enter your learning focus for this week: ").strip()
            self.data["stats"]["focus_area"] = new_focus
            print(f"✅ Focus set to: {new_focus}")
        
        if input("\nSet new weekly goals? (y/n): ").lower() == 'y':
            goals = []
            print("\nEnter your weekly goals (press Enter with empty line to finish):")
            for i in range(1, 8):  # Max 7 daily goals
                goal = input(f"Day {i} goal: ").strip()
                if not goal:
                    break
                goals.append(goal)
            
            self.data["stats"]["weekly_goals"] = goals
            print(f"✅ Set {len(goals)} weekly goals!")
        
        self._save_progress(self.data)
    
    def _update_study_streak(self):
        """Update study streak"""
        print("\n🔥 STUDY STREAK UPDATE")
        print("="*30)
        
        stats = self.data["stats"]
        today = datetime.date.today().isoformat()
        last_study = stats.get("last_study_date", "")
        
        print(f"Current Streak: {stats['current_streak']} days")
        print(f"Longest Streak: {stats['longest_streak']} days") 
        print(f"Last Study Date: {last_study or 'Never'}")
        
        if input(f"\nDid you study today ({today})? (y/n): ").lower() == 'y':
            if last_study == today:
                print("✅ Already recorded study for today!")
                return
            
            # Check if streak continues
            yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
            
            if last_study == yesterday:
                stats["current_streak"] += 1
            elif last_study == today:
                pass  # Same day, no change
            else:
                stats["current_streak"] = 1  # Reset streak
            
            stats["last_study_date"] = today
            stats["total_study_days"] += 1
            
            # Update longest streak
            if stats["current_streak"] > stats["longest_streak"]:
                stats["longest_streak"] = stats["current_streak"]
                print(f"🏆 NEW RECORD! Longest streak: {stats['longest_streak']} days!")
            
            print(f"🔥 Current streak: {stats['current_streak']} days")
            
            # Check for streak achievements  
            self._check_streak_achievements(stats["current_streak"])
            
            self._save_progress(self.data)
    
    def _check_streak_achievements(self, streak: int):
        """Check and award streak-based achievements"""
        streak_milestones = [
            (7, "Week Warrior"),
            (30, "Monthly Master"),
            (100, "Century Scholar"), 
            (365, "Year-long Learner")
        ]
        
        for milestone, name in streak_milestones:
            if streak >= milestone:
                print(f"🏆 STREAK MILESTONE: {name} ({milestone} days)!")
    
    def _export_progress(self):
        """Export progress to readable format"""
        print("\n📤 EXPORT PROGRESS")
        print("="*30)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"ml_progress_report_{timestamp}.txt"
        
        with open(export_file, 'w') as f:
            f.write("ML LEARNING PROGRESS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Overall progress
            f.write(f"Overall Completion: {self._calculate_overall_completion():.1f}%\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Algorithm progress  
            f.write("ALGORITHM MASTERY:\n")
            f.write("-" * 30 + "\n")
            for algo, data in self.data["algorithms"].items():
                f.write(f"{algo}: {data['completion_percentage']}%\n")
            
            # Badges earned
            f.write("\nBADGES EARNED:\n")
            f.write("-" * 30 + "\n")
            for ach in self.data["achievements"].values():
                if ach["earned"]:
                    f.write(f"🏆 {ach['name']} ({ach['tier']})\n")
        
        print(f"✅ Progress exported to: {export_file}")

def main():
    """Main function"""
    try:
        tracker = ProgressTracker()
        tracker.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Progress tracking cancelled. Your data is saved!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()