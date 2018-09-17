# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'



#===== Custo Patoin =====#
# Must be executed before .bash_aliases

# Utility folder

export PZ_UTILITIES="/home/pat/utilities"
export PZ_WORKSPACE="/home/pat/workspace"

# Gcloud CLI & utilities
# export GOOGLE_APPLICATION_CREDENTIALS="/home/pat/workspace/machine_controller_integration/credentials/cloud_service_account_details.json" # Using gcloud auth application-default login is better because it authorizes several loging in to several accounts

# The next line updates PATH for the Google Cloud SDK.
if [ -f "$PZ_UTILITIES/google-cloud-sdk/path.bash.inc" ]; then source "$PZ_UTILITIES/google-cloud-sdk/path.bash.inc"; fi

# The next line enables shell command completion for gcloud.
if [ -f "$PZ_UTILITIES/google-cloud-sdk/completion.bash.inc" ]; then source "$PZ_UTILITIES/google-cloud-sdk/completion.bash.inc"; fi

# Paths
export HADOOP_HOME="$PZ_UTILITIES/hadoop-2.8.2"
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="$PATH:$HADOOP_HOME/bin"
export PATH="$PZ_UTILITIES/pycharm-community-2018.1.1/bin:$PATH"  # Pycharm
export PATH="$PZ_UTILITIES/idea-IC-181.4445.78/bin:$PATH"  # IntelliJ


export PYTHONPATH="$PZ_WORKSPACE/anaximander/:$PYTHONPATH"
export PYTHONPATH="$PZ_WORKSPACE/anaximander3/:$PYTHONPATH"
export PYTHONPATH="$PZ_WORKSPACE/dataforge/:$PYTHONPATH"
export PYTHONPATH="$PZ_WORKSPACE/dataforge2/:$PYTHONPATH"
export PYTHONPATH="$PZ_WORKSPACE/pipeline/:$PYTHONPATH"
export PYTHONPATH="$PZ_WORKSPACE/python/:$PYTHONPATH"

# Spark streaming pubsub connector
export SPARK_PUBSUB_JAR="$PZ_UTILITIES/spark-pubsub/java/target/spark_pubsub-1.0-SNAPSHOT.jar"
export SPARK_PUBSUB_PYTHON_EGG="$PZ_UTILITIES/spark-pubsub/python/dist/spark_pubsub-1.0.0-py3.6.egg"

# Xtenza toolchain
export PATH="$PATH:$HOME/esp/xtensa-esp32-elf/bin"

#===== End Custo Patoin =====#


# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi


