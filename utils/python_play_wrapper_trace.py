import sys
import os
from pathlib import Path
import numpy as np
import importlib
import yaml
from utils.feature_utils import mask_features, auto_generate_mask
from flask import Flask, render_template_string
import threading
import webbrowser
import time

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree Code Execution Trace</title>
    <style>
        body { 
            margin: 0; 
            padding: 20px; 
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        .code-container {
            background-color: #1e1e1e;
            overflow: auto;
            height: 90vh;
            border: 1px solid #333;
            border-radius: 4px;
            position: relative;
        }
        pre { 
            margin: 0;
            padding: 0;
            font-size: 14px;
            line-height: 1.5;
            tab-size: 4;
        }
        .line {
            padding: 0 10px;
            white-space: pre;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        .line:hover {
            background-color: #2d2d2d;
        }
        .line-number {
            color: #858585;
            margin-right: 2em;
            user-select: none;
            min-width: 3ch;
            text-align: right;
            display: inline-block;
        }
        .highlight {
            background-color: #264f78 !important;
        }
        .keyword { color: #569cd6; }
        .string { color: #ce9178; }
        .number { color: #b5cea8; }
        .function { color: #dcdcaa; }
        .class { color: #4ec9b0; }
        .operator { color: #d4d4d4; }
    </style>
    <script>
        let currentLine = null;
        
        function updateHighlight(lineNum) {
            if (currentLine === lineNum) return;
            
            // remove the old highlight
            if (currentLine) {
                const oldLine = document.querySelector(`#L${currentLine}`);
                if (oldLine) oldLine.classList.remove('highlight');
            }
            
            // add the new highlight
            const line = document.querySelector(`#L${lineNum}`);
            if (line) {
                line.classList.add('highlight');
                line.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
                currentLine = lineNum;
            }
        }

        // check and update every 100ms
        setInterval(() => {
            fetch('/current_line')
                .then(response => response.json())
                .then(data => {
                    if (data.line) {
                        updateHighlight(data.line);
                    }
                })
                .catch(console.error);
        }, 100);
    </script>
</head>
<body>
    <div class="code-container">
        <pre>{{ code_html | safe }}</pre>
    </div>
</body>
</html>
"""

def highlight_python_code(code):
    """simple Python code syntax highlighting"""
    import re
    
    # replace HTML special characters
    code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # highlight keywords
    keywords = ['if', 'else', 'return']
    for keyword in keywords:
        code = re.sub(f'\\b{keyword}\\b', f'<span class="keyword">{keyword}</span>', code)
    
    # highlight numbers (only process standalone numbers)
    code = re.sub(r'(?<=[^a-zA-Z\d_])(-?\d+\.?\d*)(?=[^a-zA-Z\d_]|$)', 
                 r'<span class="number">\1</span>', 
                 code)
    
    # highlight operators (sort by length, ensure longer operators are processed first)
    operators = ['<=', '>=', '==', '!=']
    for op in operators:
        code = code.replace(op, f' {op} ')  # add spaces
        code = code.replace('  ', ' ')  # remove duplicate spaces
        code = code.replace(f' {op} ', f'<span class="operator">{op}</span>')
    
    return code

class PythonFunctionWrapperTrace:
    """
    Python function wrapper with code tracing functionality
    """
    def __init__(self, file_path, ff_file=None, feature_descriptions=None):
        self.file_path = file_path
        self.play_function = None
        self._ff_file = ff_file
        self._mask_indices = None
        self._feature_descriptions = feature_descriptions
        self.current_line = None
        self._is_viper = False
        self.app = Flask(__name__)
        self.setup_trace_window()
        self.load_function()
        self._setup_masking()
    
    def setup_trace_window(self):
        # read the code file
        if isinstance(self.file_path, Path):
            file_path = self.file_path
        else:
            file_path = Path(self.file_path)
        
        with open(file_path, 'r') as f:
            self.code = f.readlines()
        
        # prepare the HTML formatted code
        code_html = []
        for i, line in enumerate(self.code, 1):
            # calculate the indentation level (using the actual width of the tab)
            raw_line = line.rstrip('\n')
            indent = len(raw_line) - len(raw_line.lstrip())
            # keep the original indentation, use &nbsp; to ensure the spaces are displayed correctly in HTML
            indented_line = '&nbsp;' * indent + highlight_python_code(raw_line.lstrip())
            code_html.append(
                f'<div class="line" id="L{i}">'
                f'<span class="line-number">{i}</span>{indented_line}'
                f'</div>'
            )
        self.code_html = ''.join(code_html)
        
        # set the Flask routes
        @self.app.route('/')
        def home():
            return render_template_string(HTML_TEMPLATE, code_html=self.code_html)
        
        @self.app.route('/current_line')
        def get_current_line():
            return {'line': self.current_line}
        
        # start the Flask server in a new thread
        def run_server():
            self.app.run(port=5050, host='localhost')
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # wait for the server to start
        time.sleep(1)
        
        # open the browser
        webbrowser.open('http://localhost:5050')
    
    def trace_calls(self, frame, event, arg):
        if event != 'line':
            return self.trace_calls
        
        filename = frame.f_code.co_filename
        self.current_line = frame.f_lineno
        
        return self.trace_calls
    
    def _setup_masking(self):
        """set the feature mask, if the configuration file does not exist or the FEATURE_MASK field is not found, auto generate the mask"""
        try:
            if self._ff_file:
                with open(self._ff_file, 'r') as f:
                    config = yaml.safe_load(f)
                self._mask_indices = config.get('FEATURE_MASK', {}).get('keep_indices', None)
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Failed to load feature mask from {self._ff_file}")
            
        if self._mask_indices is None and self._feature_descriptions is not None:
            try:
                self._mask_indices = auto_generate_mask(self._feature_descriptions)
                print("Generated feature mask automatically")
            except Exception as e:
                print(f"Failed to generate feature mask: {e}")
    
    def predict(self, obs, deterministic=True):
        if not self._is_viper:
            obs = mask_features(obs, self._mask_indices)
        state = obs[0]
        
        # set the tracing
        sys.settrace(self.trace_calls)
        result = self.play_function(state)
        sys.settrace(None)
        
        return np.array([result]), None
    
    def load_function(self):
        if os.path.isfile(self.file_path):
            if str(self.file_path).endswith('.py'):
                module_name = os.path.splitext(os.path.basename(self.file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, self.file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.play_function = module.play
                if "viper" in module_name:
                    self._is_viper = True
                print("Loaded function from " + str(module_name+".py"))
            else:
                raise ValueError("The file is not a python file")
        else:
            file_list = [f for f in os.listdir(self.file_path) if f.endswith('.py')]
            file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.file_path, x)))
            module_name = os.path.splitext(file_list[-1])[0]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(self.file_path, module_name+".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.play_function = module.play
            if "viper" in module_name:
                self._is_viper = True
            print("Loaded function from " + str(module_name+".py")) 