"""
Automated setup and run script for the sentiment trading project
"""
import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'src']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"📁 Directory already exists: {dir_name}")

def check_data_file():
    """Check if the sample data file exists"""
    data_file = os.path.join('data', 'sample_news_data.csv')
    if os.path.exists(data_file):
        print("✅ Sample data file found")
        return True
    else:
        print("❌ Sample data file not found!")
        print("Please ensure 'data/sample_news_data.csv' exists")
        return False

def run_main_script():
    """Run the main sentiment analysis script"""
    print("\n" + "="*60)
    print("🚀 RUNNING SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error running main script")
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n" + "="*60)
    print("🌐 LAUNCHING STREAMLIT DASHBOARD")
    print("="*60)
    print("🔗 Dashboard will open in your web browser...")
    print("💡 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except subprocess.CalledProcessError:
        print("❌ Error launching dashboard")
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard stopped by user")

def main():
    print("🚀 SENTIMENT TRADING PROJECT SETUP")
    print("="*50)
    
    # Create directories
    print("\n1. Setting up project structure...")
    create_directories()
    
    # Check data file
    print("\n2. Checking sample data...")
    if not check_data_file():
        print("⚠️ Please add the sample data file before proceeding")
        return
    
    # Install requirements
    print("\n3. Installing dependencies...")
    if not install_requirements():
        print("⚠️ Please install requirements manually before proceeding")
        return
    
    # Ask user what to run
    print("\n" + "="*50)
    print("🎯 CHOOSE EXECUTION MODE")
    print("="*50)
    print("1. 🖥️  Run Console Version (main.py)")
    print("2. 🌐 Launch Web Dashboard (dashboard.py)")
    print("3. 🚀 Run Both (Console first, then Dashboard)")
    print("4. 🛠️  Setup Only (Don't run anything)")
    
    while True:
        choice = input("\n🔤 Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            run_main_script()
            break
        elif choice == "2":
            run_dashboard()
            break
        elif choice == "3":
            run_main_script()
            input("\n⏸️ Press Enter to launch the dashboard...")
            run_dashboard()
            break
        elif choice == "4":
            print("✅ Setup completed! You can now run:")
            print("   🖥️  Console: python main.py")
            print("   🌐 Dashboard: streamlit run dashboard.py")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
