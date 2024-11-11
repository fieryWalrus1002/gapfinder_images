# Installation of gapfinder

gapfinder is a set of python scripts to help analyze grana assembly in chloroplasts. This repository contains the scripts and data used in the paper "To Be Determined" by [An Author](https://www.madeup-paper-author.com/). These scripts are designed to be used with Jupyter Notebooks for interactive data analysis.

First we need to set up your environment, then we will install this package. There are separate instuctions depending on if you are using a Mac or Windows.

## What software is needed to run gapfinder?

In order to use gapfinder, we need to install three pieces of software. These are Python, Jupyter Notebook, and Git. Once these are installed, you can clone the gapfinder repository, open the notebooks in Jupyter, and run the scripts.

Python is a programming language, which was used to write the gapfinder scripts. We need it available on your computer to run the scripts.

Jupyter Notebook is an interactive environment for running Python code, sort of like using Word to write a document. We will use Jupyter to run the gapfinder scripts and analyze the data.

Git is a version control system that helps manage changes to the code. Think of it like a hard drive that stores different versions of your work, that you can always go back to and get a particular version, even if you've made changes.

On Mac, there is a fourth software that we need: brew. Brew is a package manager for Mac that helps install software. We will use brew to install Python, Jupyter Notebook, and Git. Its like an app store that runs in the terminal.

## Preparing the Mac environment for gapfinder

Here is a set of instructions to install Python, Jupyter Notebook, and Git on a Mac:

### Step 1: Install Homebrew (if not already installed)

Homebrew is a package manager for macOS, which simplifies the installation of software.

1. Open the Terminal (you can find it in Applications > Utilities > Terminal).
2. Install Homebrew by running the following command:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. After installation, add Homebrew to your PATH by running the following commands. This won't be exact, it will dependon your installation, so check the output from the previous command:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

### Step 2: Install Python

Homebrew can be used to install Python.

1. In the Terminal, run the following command:
   ```bash
   brew install python
   ```
2. After installation, verify the installation by checking the Python version:
   ```bash
   python3 --version
   ```

### Step 3: Install Jupyter Notebook

You can install Jupyter Notebook using `pip`, which is included with the Python installation.

1. In the Terminal, run:
   ```bash
   pip3 install notebook
   ```
2. After installation, launch Jupyter Notebook by running:
   ```bash
   jupyter notebook
   ```
3. This will open Jupyter in your default web browser.

### Step 3:

#### Option 1: Install Jupyter Notebook

You can install Jupyter Notebook using `pip`, which is included with the Python installation.

1. In the Terminal, run:
   ```bash
   pip3 install notebook
   ```
2. After installation, launch Jupyter Notebook by running:
   ```bash
   jupyter notebook
   ```
3. This will open Jupyter in your default web browser.

### Option 2: Install VSCode

If you'd rather use Microsoft's VSCode, an integrated development environment, it is quite nice for working with notebooks and provides a GUI for some of the other steps.

1. Go to (the VSCode download page)[https://code.visualstudio.com/download] and get your version.
2. Install it, and open it.
3. Install the Python extension by clicking on the extensions icon on the left side of the window, searching for Python, and clicking install.

### Step 4: Install Git

Git can also be installed via Homebrew.

1. In the Terminal, run the following command:
   ```bash
   brew install git
   ```
2. Verify the Git installation by checking the version:
   ```bash
   git --version
   ```

### Step 5: Configure Git, if needed

1. Set your Git username:
   ```bash
   git config --global user.name "Your Name"
   ```
2. Set your Git email:
   ```bash
   git config --global user.email "youremail@example.com"
   ```

Here are the next steps for installing the **gapfinder** repository on a Mac after setting up the environment, specifically using `pip` to install the repository as an editable package with the `-e` flag:

---

### Step 6: Install tesseract using brew

1. **Install tesseract using brew**:

   - In the Terminal, run the following command:
     ```bash
     brew install tesseract
     ```

---

### Step 7: Clone the gapfinder Repository

1. **Open Terminal**:

   - Press `Cmd + Space`, type "Terminal", and press Enter.

2. **Navigate to the Folder Where You Want to Clone the Repository**:

   - If you want to clone the repository into a specific folder, use the `cd` command to navigate to that location. For example, to navigate to your "Documents" folder, type:
     ```bash
     cd ~/Documents
     ```

3. **Clone the gapfinder Repository**:

   - Run the following command to clone the gapfinder repository:
     ```bash
     git clone https://github.com/fieryWalrus1002/gapfinder_images.git
     ```
   - This will download the repository into a new folder named `gapfinder_images`.

4. **Navigate into the Cloned Repository Folder**:
   - Change directory to the newly cloned folder:
     ```bash
     cd gapfinder_images
     ```

---

### Step 8: Install Dependencies Using pip

1. **Install Dependencies from `requirements.txt`**:

   - Inside the cloned repository, there is a `requirements.txt` file that lists the necessary Python packages for gapfinder.
   - To install the dependencies, run the following command:
     ```bash
     pip3 install -r requirements.txt
     ```

2. **Install the gapfinder Package in Editable Mode**:
   - In order to install gapfinder in a way that allows you to easily modify the code, we will use the `-e` flag with `pip`.
   - Run the following command to install gapfinder as an editable package:
     ```bash
     pip3 install -e .
     ```
   - The `.` at the end means "install the package from the current directory", and the `-e` flag allows you to edit the code without needing to reinstall the package every time you make changes.

---

### Step 9: Run Jupyter Notebook

1. **Launch Jupyter Notebook**:

   - After successfully installing the dependencies and gapfinder, launch Jupyter Notebook by typing:
     ```bash
     jupyter notebook
     ```
   - This will open Jupyter Notebook in your default web browser.

2. **Navigate to the Notebooks**:
   - In Jupyter, navigate to the folder containing the gapfinder notebooks and open the one you wish to run. The relative paths to the data files will work as long as you stay in the cloned repository folder.

---

### Recap of Commands for Installation

Here’s a summary of the commands you’ll use after setting up Python, Jupyter, and Git:

```bash
# Step 6: Clone the repository
git clone https://github.com/fieryWalrus1002/gapfinder_images.git
cd gapfinder_images

# Step 7: Install dependencies and install the package in editable mode
pip3 install -r requirements.txt
pip3 install -e .

# Step 8: Run Jupyter Notebook
jupyter notebook
```

This will allow you to clone the repository, install the necessary dependencies, and open Jupyter to run the gapfinder notebooks.

## Preparing the Windows Environment for gapfinder

### Step 1: Install Python

1. **Download Python:**

   - Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/).
   - Click the **"Download Python"** button. It will download an installer for Python.

2. **Run the Python Installer:**

   - Open the downloaded file to start the installation.
   - **Important:** On the first screen of the installer, check the box that says **"Add Python to PATH"** (this is crucial for the following steps).
   - Click **"Install Now"** and wait for the installation to complete.

3. **Verify Installation:**
   - Open the **Command Prompt** by pressing `Windows Key + R`, typing `cmd`, and pressing Enter.
   - In the Command Prompt, type the following command to check that Python is installed correctly:
     ```bash
     python --version
     ```
   - This should display the installed version of Python.

---

### Step 2: Install Jupyter Notebook

1. **Open Command Prompt:**

   - Open the **Command Prompt** by pressing `Windows Key + R`, typing `cmd`, and pressing Enter.

2. **Install Jupyter Notebook Using pip:**

   - Python comes with a package manager called **pip**. Use this to install Jupyter Notebook by typing:
     ```bash
     pip install notebook
     ```

3. **Run Jupyter Notebook:**
   - Once the installation finishes, you can launch Jupyter Notebook by typing the following command:
     ```bash
     jupyter notebook
     ```
   - This will open Jupyter Notebook in your web browser.

---

#### Option 2: Install VSCode

If you'd rather use Microsoft's VSCode, an integrated development environment, it is quite nice for working with notebooks and provides a GUI for some of the other steps.

1. Go to (the VSCode download page)[https://code.visualstudio.com/download] and get your version.
2. Install it, and open it.
3. Install the Python extension by clicking on the extensions icon on the left side of the window, searching for Python, and clicking install.

### Step 3: Install Git

1. **Download Git:**

   - Go to the official Git website: [https://git-scm.com/](https://git-scm.com/).
   - Click **"Download for Windows"** and run the installer.

2. **Run the Git Installer:**

   - During the installation, accept the default settings by clicking **Next** through the various screens.
   - Once the installation is finished, Git will be ready to use.

3. **Verify Git Installation:**
   - Open the **Command Prompt** and type the following to verify Git is installed:
     ```bash
     git --version
     ```
   - You should see the Git version number displayed.

---

### Step 4: Configure Git

1. **Open Git Bash:**

   - After installing Git, a program called **Git Bash** is also installed. You can use this to configure Git settings and use Git commands.

2. **Set Your Git Username and Email:**
   - Open **Git Bash** (find it by typing "Git Bash" in the Start Menu).
   - Configure Git with your username and email:
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "youremail@example.com"
     ```

---

### Step 5: Install tesseract

1. **Install tesseract using the installer:**

   - Download the installer from the following link: [https://github.com/UB-Mannheim/tesseract/wiki](UB Mannheim tesseract github page)
   - Run the installer and follow the instructions.
   - You may need to add it to your PATH, which can be done during the installation process.

---

### Step 5: Clone the gapfinder Repository

1. **Open Command Prompt or Git Bash:**

   - Navigate to the folder where you want to clone the gapfinder repository. For example, if you want to clone it into your Documents folder, type:
     ```bash
     cd C:\Users\YourUsername\Documents
     ```

2. **Clone the gapfinder Repository:**

   - Run the following command to clone the gapfinder repository:
     ```bash
     git clone https://github.com/fieryWalrus1002/gapfinder_images.git
     ```
   - This will create a new folder named `gapfinder_images` in your current directory.

3. **Navigate into the Cloned Folder:**
   - To move into the cloned repository folder, type:
     ```bash
     cd gapfinder_images
     ```

---

### Step 6: Install Dependencies and Install gapfinder

1. **Install Dependencies from `requirements.txt`:**

   - Inside the cloned repository, there’s a file called `requirements.txt` that lists all necessary Python packages for gapfinder.
   - In the Command Prompt (or Git Bash), type:
     ```bash
     pip install -r requirements.txt
     ```

2. **Install gapfinder in Editable Mode:**
   - This allows you to modify the code without reinstalling the package every time you make changes. To install gapfinder in editable mode, type:
     ```bash
     pip install -e .
     ```
   - The `.` indicates that you’re installing from the current directory.

---

### Step 7: Run Jupyter Notebook

1. **Launch Jupyter Notebook:**

   - In the Command Prompt (or Git Bash), make sure you are inside the gapfinder repository folder:
     ```bash
     cd gapfinder_images
     ```
   - Then, start Jupyter Notebook by typing:
     ```bash
     jupyter notebook
     ```

2. **Open the Notebooks:**
   - Jupyter Notebook will open in your browser, showing the folder contents. Click on the notebook file you wish to run, and start analyzing the data using the scripts.

---

### Recap of Commands for Installation on Windows

Here’s a summary of the commands you’ll use after setting up Python, Jupyter, and Git on your Windows PC:

```bash
# Step 5: Clone the repository
git clone https://github.com/fieryWalrus1002/gapfinder_images.git
cd gapfinder_images

# Step 6: Install dependencies and install gapfinder in editable mode
pip install -r requirements.txt
pip install -e .

# Step 7: Run Jupyter Notebook
jupyter notebook
```

---

By following these steps, you’ll be able to clone the **gapfinder** repository, install the required dependencies, and start working with the notebooks in Jupyter on a Windows PC.

Let me know if you need any further clarification or assistance!

### How to create a virtual environment on mac:

Here’s a summary of the commands you’ll use in gapfinder_images folder after setting up Python, Jupyter, and Git on your Mac:

```bash

# Step 1: changes to the gapfinder_images folder
cd ~/Desktop/HELMUT-DATEN-WSU/MagnusWood/Gapfinder/gapfinder_images

# Step 2: Create a virtual environment in the project folder
python3 -m venv .venv

# Step 3: Activate the virtual environment
source .venv/bin/activate

# Step 4:install notebook
pip3 install notebook

# Step 5: Install dependencies from requirements.txt
pip3 install -r requirements.txt

# Step 6: Install the gapfinder package in editable mode
pip3 install -e .

```

---
