import sys
from pathlib import Path
# Dynamically add the 'src' folder to the Python path so it works without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from eis_pem.frontend import identify_parameters_robust
from eis_pem.seis_model import all_seis_parameter_specs

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')
        self.widget.update_idletasks()
        
    def flush(self):
        pass

class EISPemGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EIS-PEM Parameter Identification (GUI Version)")
        self.geometry("800x600")
        self.configure(padx=20, pady=20)
        
        # Style
        style = ttk.Style(self)
        style.theme_use('clam')
        
        self.input_file = tk.StringVar()
        
        self._build_ui()
        
    def _build_ui(self):
        # --- File Selection Frame ---
        frame_file = ttk.LabelFrame(self, text=" 1. Select Input Data (CSV) ", padding=10)
        frame_file.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(frame_file, text="Input CSV Path:").pack(side=tk.LEFT)
        entry_file = ttk.Entry(frame_file, textvariable=self.input_file, width=50)
        entry_file.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        btn_browse = ttk.Button(frame_file, text="Browse...", command=self.browse_file)
        btn_browse.pack(side=tk.LEFT)
        
        # --- Parameters Frame ---
        frame_params = ttk.LabelFrame(self, text=" 2. Configuration ", padding=10)
        frame_params.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(frame_params, text="Expected Noise Level (e.g., 0.01 for 1%):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.noise_level = tk.DoubleVar(value=0.03)
        ttk.Entry(frame_params, textvariable=self.noise_level, width=10).grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(frame_params, text="Random Starts (n_starts):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.n_starts = tk.IntVar(value=3)
        ttk.Entry(frame_params, textvariable=self.n_starts, width=10).grid(row=1, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(frame_params, text="Error Strictness (max_ci95, e.g., 0.5 for 50%):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_ci95 = tk.DoubleVar(value=0.50)
        ttk.Entry(frame_params, textvariable=self.max_ci95, width=10).grid(row=2, column=1, sticky=tk.W, padx=10)
        
        # --- Run Button ---
        self.btn_run = ttk.Button(self, text="▶ RUN IDENTIFICATION", command=self.run_pipeline)
        self.btn_run.pack(pady=(0, 15), fill=tk.X, ipady=10)
        
        # --- Logs Frame ---
        frame_logs = ttk.LabelFrame(self, text=" 3. Execution Logs ", padding=10)
        frame_logs.pack(fill=tk.BOTH, expand=True)
        
        self.text_log = tk.Text(frame_logs, state='disabled', bg='black', fg='white', font=("Consolas", 12))
        scrollbar = ttk.Scrollbar(frame_logs, command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=scrollbar.set)
        
        self.text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Redirect stdout and stderr
        sys.stdout = TextRedirector(self.text_log, "stdout")
        sys.stderr = TextRedirector(self.text_log, "stderr")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input CSV",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if filename:
            self.input_file.set(filename)

    def run_pipeline(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input CSV file first.")
            return
            
        self.btn_run.config(state=tk.DISABLED, text="Running... Please wait.")
        self.text_log.configure(state='normal')
        self.text_log.delete(1.0, tk.END)
        self.text_log.configure(state='disabled')
        
        # Run in a separate thread so GUI doesn't freeze
        thread = threading.Thread(target=self._worker)
        thread.daemon = True
        thread.start()

    def _worker(self):
        try:
            print("--- Starting EIS-PEM Pipeline ---")
            csv_path = self.input_file.get()
            print(f"Reading data from: {csv_path}")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Validate Columns
            required_cols = {'Frequency_Hz', 'Re_Z', 'Im_Z', 'Temperature_K', 'SOC'}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in CSV: {missing}")
                
            # Find unique conditions preserving order
            conditions_df = df[['Temperature_K', 'SOC']].drop_duplicates()
            conditions_list = []
            for _, row in conditions_df.iterrows():
                conditions_list.append({
                    "temperature_K": float(row['Temperature_K']),
                    "SOC": float(row['SOC'])
                })
                
            # Build Request Dictionary
            request = {
                "freq_hz": df["Frequency_Hz"].tolist(),
                "z_obs_real": df["Re_Z"].tolist(),
                "z_obs_imag": df["Im_Z"].tolist(),
                "conditions": conditions_list,
                "strategy": "adaptive",
                "noise_level": self.noise_level.get(),
                "n_starts": self.n_starts.get(),
                "max_ci95": self.max_ci95.get(),
                # Turn on the data cleaner explicitly
                "filter_config": {"enable_kramers_kronig": True}
            }
            
            print("Data loaded successfully. Starting Identifiability Analysis and Optimization...")
            result = identify_parameters_robust(request)
            
            print("\nOptimization Complete! Exporting to Excel...")
            out_path = csv_path.replace('.csv', '_Results.xlsx')
            self._export_excel(result, out_path)
            
            print(f"\n--- SUCCESS ---")
            print(f"Results perfectly saved to:\n{out_path}")
            messagebox.showinfo("Success", f"Identification completed successfully!\nResults saved to: {out_path}")
            
        except Exception as e:
            print("\n--- ERROR ---")
            print(f"An error occurred: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Execution Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.btn_run.config(state=tk.NORMAL, text="▶ RUN IDENTIFICATION")

    def _export_excel(self, result_dict, out_path):
        theta_best = result_dict.get('theta_best', {})
        free_names = result_dict.get('free_parameters', [])
        fixed_dict = result_dict.get('fixed_parameters', {})
        diagnostics_list = result_dict.get('diagnostics', [])
        
        # Build Parameter DataFrame
        specs = all_seis_parameter_specs()
        param_data = []
        
        # Map parameter names to their diagnostics for quick lookup
        diag_map = {d['name']: d for d in diagnostics_list}
        
        for s in specs:
            name = s.name
            gt = s.initial_value
            if name in theta_best:
                val = theta_best[name]
                if name in free_names:
                    # It was fitted
                    status = "Fitted"
                    ci_abs = "N/A"
                    if name in diag_map and 'ci95_relative' in diag_map[name]:
                        ci_abs = diag_map[name]['ci95_relative'] * val
                else:
                    # It was fixed
                    status = "Fixed"
                    ci_abs = "N/A"
                    
                param_data.append({
                    "Parameter": name,
                    "Initial/Reference": gt,
                    "Final_Value": val,
                    "CI95_Absolute": ci_abs,
                    "Status": status
                })
                
        df_params = pd.DataFrame(param_data)
        
        # Build Diagnostics DataFrame
        scalar_keys = ['final_cost', 'quality_score', 'n_points_used', 'n_points_total', 'rank', 'effective_rank', 'condition_number']
        diag_data = []
        for k in scalar_keys:
            if k in result_dict:
                diag_data.append({"Metric": k, "Value": result_dict[k]})
                
        df_diag = pd.DataFrame(diag_data)
        
        try:
            with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                df_params.to_excel(writer, sheet_name='Parameters', index=False)
                df_diag.to_excel(writer, sheet_name='Summary_Diagnostics', index=False)
        except ImportError:
            # Fallback to CSV if openpyxl is missing
            csv_path = out_path.replace('.xlsx', '_Parameters.csv')
            df_params.to_csv(csv_path, index=False)
            messagebox.showwarning(
                "Missing Dependency",
                "Python module 'openpyxl' is not installed in your current environment.\n\n"
                "To output directly to .xlsx, please run:\n"
                "pip install openpyxl\n\n"
                f"For now, the parameters have been saved as a CSV file at:\n{csv_path}"
            )

if __name__ == "__main__":
    app = EISPemGUI()
    app.mainloop()
