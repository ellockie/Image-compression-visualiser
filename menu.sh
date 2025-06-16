#!/bin/sh
# Exit on errors:
set -e
# v. 0.4

# Define color variables
RED=$(tput setaf 1)    # for regular options
GREEN=$(tput setaf 2)    # for regular options
CYAN=$(tput setaf 6)     # for special options/notes
DIM=$(tput dim)
RESET=$(tput sgr0)

# Copied from:  ~/Dropbox/____Scripts/___Bash/___Project_menu

# Exit immediately if a command exits with a non-zero status.
set -e

# Error handling function
error_exit() {
    echo "  ${RED}Error: $1${RESET}" >&2
    exit 1
}

# Validate environment and configuration
validate_environment() {
    echo "Validating environment and configuration..."
    # Check environment variable
    [ -z "$VIRTUAL_ENV_DIR" ] && error_exit "VIRTUAL_ENV_DIR environment variable is not set."

    # Check configuration file
    [ ! -r .project_config.yml ] && error_exit ".project_config.yml file not found or not readable."

    # Get config variables
    project_name=$(yq eval '.project.name' .project_config.yml)
    python_version=$(yq eval '.project.python_version' .project_config.yml)
    main_filename=$(yq eval '.project.main_filename' .project_config.yml)

    # Verify all variables are non-empty
    [ -z "$project_name" ] && error_exit "project.name is not set"
    [ -z "$python_version" ] && error_exit "project.python_version is not set"
    [ -z "$main_filename" ] && error_exit "project.main_filename is not set"

    echo " "
    echo " (*)      Project name:  $project_name"
    echo " (*)    Python version:  $python_version"
    echo " (*)     Main filename:  $main_filename"
    echo " "

    # Type/format validation
    echo "$python_version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' ||
        error_exit "Invalid Python version format: $python_version. Expected format: X.Y.Z (e.g., 3.9.0)"

    echo "$main_filename" | grep -qE '\.(py|sh)$' ||
        error_exit "Invalid main filename: $main_filename. Expected extension: .py or .sh"

    # Check if the expected Python version is installed
    pyenv versions | grep -q "$python_version" ||
        error_exit "Python '$python_version' is not installed. Please install it using pyenv."
    echo "Environment and configuration validated successfully."
}

# Setup project virtual environment
setup_project() {
    validate_environment

    # 1) Ensure pyenv is initialized in THIS shell (so we can use pyenv in the script)
    if command -v pyenv >/dev/null 2>&1; then
        eval "$(pyenv init -)"
    else
        echo "pyenv not found; please install or ensure it is on PATH"
        return 1
    fi

    # Root directory of the project & venv settings
    venv_dir="$VIRTUAL_ENV_DIR/$project_name"
    activate_path="$venv_dir/bin/activate"

    echo " [] Current Python version: $(python --version)"
    echo " [] Setting local Python version to $python_version..."

    # 2) Set local Python version with pyenv
    pyenv local "$python_version"

    # 3) Re-initialize pyenv in THIS script session to pick up the newly written .python-version
    eval "$(pyenv init -)"
    echo " [] Python version set to: $(python --version)"  # Should now show the new version

    # Create virtual environment if it doesn't exist
    if [ ! -d "$venv_dir" ]; then
        python -m venv "$venv_dir"
        echo "Virtual environment created: $venv_dir"
    else
        echo "Virtual environment, $venv_dir, already exists."
    fi

    # Activate, install dependencies, then deactivate
    source "$activate_path"
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate

    echo "Project setup / updated successfully."
}


# Activate project virtual environment
activate_project() {
    # THIS SCRIPT ALSO NEEDS TO BE SOURCED! ( BUT IT CRASHES! )
    # source menu.sh
    # or
    # . menu.sh

    # This is a command built into bash and some other shells that automatically export any subsequently defined variables to the environment of child processes. Here, -a is a flag that stands for "allexport".
    # set -a

    # source is a bash shell built-in command that executes the content of the file passed as argument, in the current shell. .env is commonly used to hold a list of environment variables to be used by an application, with each line in the file being a key value pair in the form KEY=VALUE.
    # The source .env command reads the file named .env in the current directory and executes the commands in the current shell environment. Because set -a was called earlier, all variables defined in the .env file will be exported as environment variables, not just defined as shell variables.
    # You need to source the script instead of executing it. Sourcing runs the script within your current shell session, allowing any environment changes to persist after the script finishes.
    validate_environment

    venv_dir="$VIRTUAL_ENV_DIR/$project_name"
    activate_path="$venv_dir/bin/activate"

    source "$activate_path"
    echo "Virtual environment activated: $venv_dir"
}

# Run project main script
run_project() {
    validate_environment

    venv_dir="$VIRTUAL_ENV_DIR/$project_name"
    activate_path="$venv_dir/bin/activate"

    source "$activate_path"

    # Execute script based on file extension
    case "${main_filename##*.}" in
        "sh")
            # Execute shell script
            sh "$main_filename"
            ;;
        "py")
            # Execute python script
            python "$main_filename"
            ;;
        *)
            echo "${RED}Error: Unsupported file type. Only .sh and .py files are supported.${RESET}"
            deactivate
            return 1
            ;;
    esac

    deactivate
}

# Display usage information
usage() {
    echo ""
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  1 - Run Project"
    echo "  2 - Setup Project"
    echo "  3 - Activate Project"
    echo "  4 - Exit"
    echo ""
    echo "If no option is provided, an interactive menu will be shown."
    echo ""
}

# Execute selected option
execute_option() {
    case "$1" in
        1)
            run_project
            ;;
        2)
            setup_project
            ;;
        3)
            activate_project
            ;;
        4)
            printf "\n%sGoodbye!%s\n\n" "${GREEN}" "${RESET}"
            exit 0
            ;;
        *)
            printf "\n${RED}Invalid choice. Exiting.${RESET}\n\n"
            exit 1
            ;;
    esac
}

# Main script logic
if [ $# -eq 1 ]; then
    # Validate argument
    if ! echo "$1" | grep -qE '^[1-4]$'; then
        echo "${RED}Error: Invalid option. Must be a number between 1 and 4.${RESET}"
        usage
        exit 1
    fi

    # Execute the specified option
    execute_option "$1"
elif [ $# -gt 1 ]; then
    # More than one argument provided
    echo "${RED}Error: Too many arguments.${RESET}"
    usage
    exit 1
else
    # No arguments, show interactive menu
    printf "\nPlease choose an option:\n\n"
    echo "  ${RED}1. ${GREEN}Run Project${RESET}"
    echo "  ${RED}2. ${GREEN}Setup Project${RESET}"
    echo "  ${RED}3. ${GREEN}Activate Project${RESET} ${DIM}- THIS SCRIPT ALSO NEEDS TO BE SOURCED! ( but it crashes! )${RESET}"
    echo "  ${RED}4. ${GREEN}Exit${RESET}"

    printf "\nEnter choice [1-4]:  "
    read choice < /dev/tty

    execute_option "$choice"
fi
