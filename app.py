
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Salary Predictor & Data Analyzer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Create the main interface
        self.create_widgets()
        
        # Try to load default dataset
        self.load_default_dataset()
    
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading and Overview
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Overview")
        self.create_data_tab()
        
        # Tab 2: Visualizations
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Data Visualizations")
        self.create_viz_tab()
        
        # Tab 3: Model Training
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="Model Training")
        self.create_model_tab()
        
        # Tab 4: Salary Prediction
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="Salary Prediction")
        self.create_predict_tab()
    
    def create_data_tab(self):
        # File loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Load Dataset", padding=10)
        load_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load CSV File", command=self.load_file).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Load Default Dataset", command=self.load_default_dataset).pack(side='left', padx=5)
        
        # Data info section
        info_frame = ttk.LabelFrame(self.data_frame, text="Dataset Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Text widget for data info
        self.info_text = tk.Text(info_frame, height=15, wrap='word')
        scrollbar = ttk.Scrollbar(info_frame, orient='vertical', command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_viz_tab(self):
        # Visualization controls
        control_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Income Distribution", command=self.plot_income_distribution).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Education Impact", command=self.plot_education_impact).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Age Groups", command=self.plot_age_groups).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Work Hours", command=self.plot_work_hours).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Correlation Matrix", command=self.plot_correlation).pack(side='left', padx=5)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
    
    def create_model_tab(self):
        # Model training controls
        train_frame = ttk.LabelFrame(self.model_frame, text="Model Training", padding=10)
        train_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(train_frame, text="Train Model", command=self.train_model).pack(side='left', padx=5)
        ttk.Button(train_frame, text="Evaluate Model", command=self.evaluate_model).pack(side='left', padx=5)
        
        # Model results
        results_frame = ttk.LabelFrame(self.model_frame, text="Model Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, wrap='word')
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
    
    def create_predict_tab(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.predict_frame, text="Enter Your Information", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Create input fields
        self.inputs = {}
        
        # Age
        ttk.Label(input_frame, text="Age:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.inputs['age'] = ttk.Spinbox(input_frame, from_=17, to=90, width=10)
        self.inputs['age'].grid(row=0, column=1, padx=5, pady=2)
        
        # Education
        ttk.Label(input_frame, text="Education:").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.inputs['education'] = ttk.Combobox(input_frame, values=[
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
            '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
        ], width=15)
        self.inputs['education'].grid(row=0, column=3, padx=5, pady=2)
        
        # Hours per week
        ttk.Label(input_frame, text="Hours per week:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.inputs['hours_per_week'] = ttk.Spinbox(input_frame, from_=1, to=99, width=10)
        self.inputs['hours_per_week'].grid(row=1, column=1, padx=5, pady=2)
        
        # Occupation
        ttk.Label(input_frame, text="Occupation:").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        self.inputs['occupation'] = ttk.Combobox(input_frame, values=[
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv',
            'Armed-Forces'
        ], width=15)
        self.inputs['occupation'].grid(row=1, column=3, padx=5, pady=2)
        
        # Predict button
        predict_btn = ttk.Button(input_frame, text="Predict Salary", command=self.predict_salary)
        predict_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Result frame
        result_frame = ttk.LabelFrame(self.predict_frame, text="Prediction Result", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.result_label = ttk.Label(result_frame, text="Enter your information and click 'Predict Salary'", 
                                     font=('Arial', 14))
        self.result_label.pack(pady=20)
        
        # Probability frame
        self.prob_frame = ttk.Frame(result_frame)
        self.prob_frame.pack(fill='x', pady=10)
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.process_data()
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_default_dataset(self):
        try:
            # Try to load adult.csv (user should rename their 'adult 3.csv' to 'adult.csv')
            self.df = pd.read_csv('adult.csv')
            self.process_data()
            messagebox.showinfo("Success", "Default dataset (adult.csv) loaded successfully!")
        except FileNotFoundError:
            try:
                # Try alternative name
                self.df = pd.read_csv('adult 3.csv')
                self.process_data()
                messagebox.showinfo("Success", "Dataset (adult 3.csv) loaded successfully!")
            except FileNotFoundError:
                messagebox.showwarning("Warning", 
                    "Default dataset not found. Please ensure 'adult.csv' or 'adult 3.csv' is in the same directory as this script.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default dataset: {str(e)}")
    
    def process_data(self):
        if self.df is None:
            return
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Handle different possible column names for income
        income_cols = ['income', 'salary', 'class', 'target']
        income_col = None
        for col in income_cols:
            if col in self.df.columns:
                income_col = col
                break
        
        if income_col is None:
            # If no standard income column found, use the last column
            income_col = self.df.columns[-1]
        
        # Rename income column to 'income' for consistency
        if income_col != 'income':
            self.df = self.df.rename(columns={income_col: 'income'})
        
        # Clean income values
        self.df['income'] = self.df['income'].str.strip()
        
        # Remove rows with missing values
        self.df = self.df.dropna()
        
        # Display dataset info
        self.display_data_info()
    
    def display_data_info(self):
        if self.df is None:
            return
        
        info_text = f"Dataset Shape: {self.df.shape}\n\n"
        info_text += "Column Information:\n"
        info_text += "=" * 50 + "\n"
        
        for col in self.df.columns:
            info_text += f"{col}:\n"
            info_text += f"  Type: {self.df[col].dtype}\n"
            info_text += f"  Unique values: {self.df[col].nunique()}\n"
            if self.df[col].dtype == 'object':
                info_text += f"  Values: {list(self.df[col].unique())[:5]}...\n"
            info_text += "\n"
        
        info_text += "\nIncome Distribution:\n"
        info_text += "=" * 30 + "\n"
        income_counts = self.df['income'].value_counts()
        for income, count in income_counts.items():
            percentage = (count / len(self.df)) * 100
            info_text += f"{income}: {count} ({percentage:.1f}%)\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def plot_income_distribution(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        income_counts = self.df['income'].value_counts()
        colors = ['#3498db', '#e74c3c']
        
        ax.pie(income_counts.values, labels=income_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Income Distribution', fontsize=14, fontweight='bold')
        
        self.canvas.draw()
    
    def plot_education_impact(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Calculate percentage of high earners by education
        education_income = pd.crosstab(self.df['education'], self.df['income'], normalize='index') * 100
        
        if '>50K' in education_income.columns:
            high_income_col = '>50K'
        elif '>50K.' in education_income.columns:
            high_income_col = '>50K.'
        else:
            high_income_col = education_income.columns[1]  # Assume second column is high income
        
        education_income[high_income_col].plot(kind='bar', ax=ax, color='#2ecc71')
        ax.set_title('High Income Percentage by Education Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Education Level')
        ax.set_ylabel('Percentage Earning >$50K')
        plt.xticks(rotation=45, ha='right')
        
        self.canvas.draw()
    
    def plot_age_groups(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create age groups (temporary for visualization only)
        df_temp = self.df.copy()
        df_temp['age_group'] = pd.cut(df_temp['age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['17-25', '26-35', '36-45', '46-55', '56+'])
        
        age_income = pd.crosstab(df_temp['age_group'], df_temp['income'], normalize='index') * 100
        
        if '>50K' in age_income.columns:
            high_income_col = '>50K'
        elif '>50K.' in age_income.columns:
            high_income_col = '>50K.'
        else:
            high_income_col = age_income.columns[1]
        
        age_income[high_income_col].plot(kind='bar', ax=ax, color='#f39c12')
        ax.set_title('High Income Percentage by Age Group', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Percentage Earning >$50K')
        plt.xticks(rotation=0)
        
        self.canvas.draw()
    
    def plot_work_hours(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create work hours groups (temporary for visualization only)
        df_temp = self.df.copy()
        df_temp['hours_group'] = pd.cut(df_temp['hours-per-week'], 
                                       bins=[0, 35, 45, 100], 
                                       labels=['Part-time (<35h)', 'Full-time (35-45h)', 'Overtime (>45h)'])
        
        hours_income = pd.crosstab(df_temp['hours_group'], df_temp['income'], normalize='index') * 100
        
        if '>50K' in hours_income.columns:
            high_income_col = '>50K'
        elif '>50K.' in hours_income.columns:
            high_income_col = '>50K.'
        else:
            high_income_col = hours_income.columns[1]
        
        hours_income[high_income_col].plot(kind='bar', ax=ax, color='#9b59b6')
        ax.set_title('High Income Percentage by Work Hours', fontsize=14, fontweight='bold')
        ax.set_xlabel('Work Hours Category')
        ax.set_ylabel('Percentage Earning >$50K')
        plt.xticks(rotation=0)
        
        self.canvas.draw()
    
    def plot_correlation(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Not enough numeric columns for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.canvas.draw()
    
    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        try:
            # Prepare features and target (exclude temporary visualization columns)
            X = self.df.drop(['income'], axis=1)
            # Remove any temporary columns created for visualization
            temp_cols = ['age_group', 'hours_group']
            for col in temp_cols:
                if col in X.columns:
                    X = X.drop([col], axis=1)
            
            y = self.df['income']
            
            # Handle categorical variables
            self.label_encoders = {}
            X_encoded = X.copy()
            
            for column in X.columns:
                if X[column].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[column] = le.fit_transform(X[column].astype(str))
                    self.label_encoders[column] = le
            
            # Store feature columns
            self.feature_columns = X_encoded.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            results = f"Model Training Completed!\n\n"
            results += f"Accuracy: {accuracy:.4f}\n\n"
            results += "Feature Importance:\n"
            results += "=" * 30 + "\n"
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in feature_importance.iterrows():
                results += f"{row['feature']}: {row['importance']:.4f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            messagebox.showinfo("Success", f"Model trained successfully! Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def evaluate_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
        
        try:
            # Prepare data (exclude temporary visualization columns)
            X = self.df.drop(['income'], axis=1)
            # Remove any temporary columns created for visualization
            temp_cols = ['age_group', 'hours_group']
            for col in temp_cols:
                if col in X.columns:
                    X = X.drop([col], axis=1)
            
            y = self.df['income']
            
            X_encoded = X.copy()
            for column in X.columns:
                if column in self.label_encoders:
                    X_encoded[column] = self.label_encoders[column].transform(X[column].astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Generate detailed evaluation
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation = f"Model Evaluation Results\n\n"
            evaluation += f"Accuracy: {accuracy:.4f}\n\n"
            evaluation += "Classification Report:\n"
            evaluation += "=" * 40 + "\n"
            evaluation += report + "\n\n"
            evaluation += "Confusion Matrix:\n"
            evaluation += "=" * 20 + "\n"
            evaluation += str(cm)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, evaluation)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model: {str(e)}")
    
    def predict_salary(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
        
        try:
            # Get input values
            age = int(self.inputs['age'].get())
            education = self.inputs['education'].get()
            hours_per_week = int(self.inputs['hours_per_week'].get())
            occupation = self.inputs['occupation'].get()
            
            if not all([education, occupation]):
                messagebox.showwarning("Warning", "Please fill in all fields.")
                return
            
            # Create input DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'education': [education],
                'hours-per-week': [hours_per_week],
                'occupation': [occupation]
            })
            
            # Add missing columns with default values
            for col in self.feature_columns:
                if col not in input_data.columns:
                    if col in self.label_encoders:
                        # Use the most common value for categorical variables
                        most_common = self.df[col].mode()[0] if col in self.df.columns else 'Unknown'
                        input_data[col] = [most_common]
                    else:
                        # Use mean for numeric variables
                        mean_val = self.df[col].mean() if col in self.df.columns else 0
                        input_data[col] = [mean_val]
            
            # Encode categorical variables
            input_encoded = input_data.copy()
            for column in input_data.columns:
                if column in self.label_encoders:
                    try:
                        input_encoded[column] = self.label_encoders[column].transform(input_data[column].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        input_encoded[column] = [0]
            
            # Ensure all required columns exist and reorder to match training data
            for col in self.feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = [0]  # Default value for missing columns
            input_encoded = input_encoded[self.feature_columns]
            
            # Make prediction
            prediction = self.model.predict(input_encoded)[0]
            probabilities = self.model.predict_proba(input_encoded)[0]
            
            # Display result
            if prediction in ['>50K', '>50K.']:
                result_text = "🎉 Predicted Income: >$50K (High Income)"
                color = 'green'
            else:
                result_text = "📊 Predicted Income: ≤$50K (Moderate Income)"
                color = 'orange'
            
            self.result_label.config(text=result_text, foreground=color)
            
            # Clear and update probability frame
            for widget in self.prob_frame.winfo_children():
                widget.destroy()
            
            # Display probabilities
            classes = self.model.classes_
            ttk.Label(self.prob_frame, text="Prediction Confidence:", font=('Arial', 12, 'bold')).pack()
            
            for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                prob_text = f"{cls}: {prob:.2%}"
                ttk.Label(self.prob_frame, text=prob_text, font=('Arial', 10)).pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")

def main():
    root = tk.Tk()
    app = SalaryPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
