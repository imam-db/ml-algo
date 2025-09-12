#!/usr/bin/env python3
"""
Algorithm Playground Launcher
=============================

Main entry point for the ML Algorithm Playground.
Provides an interactive menu to access all playground tools.

Usage: uv run python playground_launcher.py
"""

import os
import sys
import subprocess
import argparse

class PlaygroundLauncher:
    """Interactive launcher for all playground tools"""
    
    def __init__(self):
        self.tools = {
            '1': {
                'name': 'Algorithm Racing Tool',
                'description': 'Compare multiple algorithms side-by-side',
                'script': 'algorithm_race.py',
                'icon': '🏁',
                'examples': [
                    '--quick_test',
                    '--algorithms "random_forest,svm,logistic_regression" --quick_test',
                    '--data_path your_data.csv --stats_test'
                ]
            },
            '2': {
                'name': 'Parameter Playground',
                'description': 'Interactive hyperparameter tuning with real-time visualization',
                'script': 'parameter_playground.py',
                'icon': '🎛️',
                'examples': [
                    '--algorithm random_forest --data_type classification',
                    '--algorithm svm --data_type moons',
                    '--demo'
                ]
            },
            '3': {
                'name': 'Model Visualizer',
                'description': 'Visualize decision boundaries and model behavior',
                'script': 'model_visualizer.py',
                'icon': '📊',
                'examples': [
                    '--algorithm random_forest --data_type moons',
                    '--algorithm svm --data_type circles --save visualization.png',
                    '--algorithm logistic_regression --n_samples 500'
                ]
            },
            '4': {
                'name': 'Learning Curve Explorer',
                'description': 'Analyze how algorithms learn over time',
                'script': 'learning_curves.py',
                'icon': '📈',
                'examples': [
                    '--algorithm random_forest --data_type classification',
                    '--algorithm gradient_boosting --save learning_analysis.png',
                    '--algorithm mlp --quick'
                ]
            },
            '5': {
                'name': 'Feature Engineering Lab',
                'description': 'Experiment with feature transformations',
                'script': 'feature_lab.py',
                'icon': '🧪',
                'examples': [
                    '--interactive',
                    '--demo --data_type classification',
                    '--n_features 15 --n_samples 800'
                ]
            },
            '6': {
                'name': 'Algorithm Animator',
                'description': 'Step-by-step visual animations of algorithms',
                'script': 'algorithm_animator.py',
                'icon': '🎬',
                'examples': [
                    '--algorithm kmeans --k 4',
                    '--algorithm gradient_descent --learning_rate 0.1',
                    '--algorithm knn --k 7 --save knn_animation.gif'
                ]
            },
            '7': {
                'name': 'Data Generator',
                'description': 'Create synthetic datasets for testing',
                'script': 'data_generator.py',
                'icon': '🎲',
                'examples': [
                    '--interactive',
                    '--type moons --noise 0.2 --save moon_data.csv',
                    '--type polynomial --n_samples 500'
                ]
            }
        }
    
    def show_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🎮 MACHINE LEARNING ALGORITHM PLAYGROUND")
        print("="*60)
        print("Interactive tools for exploring ML algorithms")
        print()
        
        for key, tool in self.tools.items():
            print(f"{key}. {tool['icon']} {tool['name']}")
            print(f"   {tool['description']}")
        
        print("\n8. 📚 Help & Documentation")
        print("9. ❌ Exit")
        print("\n" + "-"*60)
    
    def show_tool_help(self, tool_key):
        """Show detailed help for a specific tool"""
        if tool_key not in self.tools:
            print("❌ Invalid tool selection")
            return
        
        tool = self.tools[tool_key]
        print(f"\n{tool['icon']} {tool['name'].upper()}")
        print("="*50)
        print(f"Description: {tool['description']}")
        print(f"Script: {tool['script']}")
        print("\n💡 Example Usage:")
        
        for i, example in enumerate(tool['examples'], 1):
            print(f"  {i}. uv run python {tool['script']} {example}")
        
        print("\n🔧 Options:")
        print("  1. Run with default settings")
        print("  2. Run with custom parameters") 
        print("  3. View full help")
        print("  4. Back to main menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            self.run_tool(tool['script'])
        elif choice == '2':
            params = input("Enter parameters: ").strip()
            self.run_tool(tool['script'], params)
        elif choice == '3':
            self.run_tool(tool['script'], '--help')
        elif choice == '4':
            return
        else:
            print("❌ Invalid choice")
    
    def run_tool(self, script, params=''):
        """Run a playground tool"""
        try:
            cmd = f"uv run python {script} {params}".strip()
            print(f"\n🚀 Running: {cmd}")
            print("-" * 50)
            
            result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(__file__))
            
            if result.returncode == 0:
                print("✅ Tool completed successfully")
            else:
                print("⚠️ Tool exited with errors")
                
        except KeyboardInterrupt:
            print("\n⏹️ Tool interrupted by user")
        except Exception as e:
            print(f"❌ Error running tool: {e}")
        
        input("\nPress Enter to continue...")
    
    def show_general_help(self):
        """Show general help and documentation"""
        print("\n📚 ALGORITHM PLAYGROUND HELP")
        print("="*40)
        print()
        print("🎯 Purpose:")
        print("  The Algorithm Playground provides interactive tools for")
        print("  learning and experimenting with machine learning algorithms.")
        print()
        print("🛠️ Available Tools:")
        
        for key, tool in self.tools.items():
            print(f"  • {tool['name']}: {tool['description']}")
        
        print()
        print("🚀 Getting Started:")
        print("  1. Select a tool from the main menu")
        print("  2. Choose to run with defaults or custom parameters")
        print("  3. Follow the interactive prompts")
        print("  4. Explore and learn!")
        print()
        print("💡 Tips:")
        print("  • Start with 'Algorithm Racing' for comparisons")
        print("  • Use 'Parameter Playground' for interactive tuning") 
        print("  • Try 'Algorithm Animator' for visual learning")
        print("  • Use 'Feature Engineering Lab' for data preprocessing")
        print()
        print("📖 Documentation:")
        print("  • README.md in the playground directory")
        print("  • Individual tool help with --help flag")
        print("  • Examples in each tool's docstring")
        
        input("\nPress Enter to continue...")
    
    def launch(self):
        """Main interactive launcher loop"""
        try:
            while True:
                self.show_main_menu()
                choice = input("Select tool (1-9): ").strip()
                
                if choice in self.tools:
                    self.show_tool_help(choice)
                elif choice == '8':
                    self.show_general_help()
                elif choice == '9':
                    print("👋 Thanks for using the Algorithm Playground!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-9.")
                    input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="ML Algorithm Playground - Interactive launcher for all tools"
    )
    parser.add_argument('--tool', type=str, 
                       help='Launch specific tool directly (1-7)')
    parser.add_argument('--list', action='store_true',
                       help='List all available tools')
    
    args = parser.parse_args()
    
    launcher = PlaygroundLauncher()
    
    if args.list:
        print("Available tools:")
        for key, tool in launcher.tools.items():
            print(f"{key}. {tool['icon']} {tool['name']} - {tool['description']}")
        return
    
    if args.tool:
        if args.tool in launcher.tools:
            launcher.show_tool_help(args.tool)
        else:
            print(f"❌ Unknown tool: {args.tool}")
            print("Use --list to see available tools")
    else:
        launcher.launch()

if __name__ == "__main__":
    main()