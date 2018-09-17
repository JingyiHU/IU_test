##### ea - alias for editing aliases
#
#When setting up a new aliases file, or having creating a new file.. About every time after editing an aliases file, I source it. This alias makes editing alias a
#bit easier and they are useful right away. Note if the source failed, it will not echo "aliases sourced".
#
#Sub in gedit for your favorite editor, or alter for ksh, sh, etc.
#
alias ea='gedit ~/.bash_aliases; source ~/.bash_aliases && source $HOME/.bash_aliases && echo "aliases sourced  --ok."'
#

# Conda Env shortcuts
alias pipeline_start="source activate pipeline; cd $PZ_WORKSPACE/pipeline"
alias dataforge2_start="source activate pipeline2; cd $PZ_WORKSPACE/dataforge2"
alias gateway_start="source activate gateway; cd $PZ_WORKSPACE/gateway/gateway_administration"
alias firmware_start="source activate firmware; cd $PZ_WORKSPACE/productivity"
alias tzigane_start="source activate tzigane; cd $PZ_WORKSPACE/tzigane; export TZIGANE_DOMAIN=localhost:5006; export GCLOUD_PROJECT_ID='$(gcloud config get-value project -q)'"
alias pyspark_start="source activate pyspark_env; cd $PZ_WORKSPACE/pyspark;"

alias gohome="source deactivate; cd"

alias conda_install_requirements="while read requirement; do conda install --yes $requirement; done < requirements.txt 2>error.log"

# Google Cloud shortcuts
alias gcloud_test="gcloud config configurations activate default"
alias gcloud_prod="gcloud config configurations activate prod"
alias gcloud_v2="gcloud config configurations activate iutportal"
alias sql_proxy="$PZ_UTILITIES/cloud_sql_proxy -instances=infinite-uptime-1232:us-central1:df2=tcp:3305,infinite-uptime-1232:us-central1:server2=tcp:3306,infinite-uptime-1232:us-central1:gateway=tcp:3307,iutportal:us-central1:df2=tcp:3308,"

# Django shortcuts
alias django_shell="python manage.py shell"
alias django_server="python manage.py runserver"

# Git shortcuts
alias gs="git status"

# Other shortcuts
alias mqtt_spy="cd ~/mqtt-spy; java -jar ~/mqtt-spy/mqtt-spy-1.0.0.jar"

# Use jupyter for pyspark
alias pyspark_use_jupyter="export PYSPARK_DRIVER_PYTHON=jupyter; export PYSPARK_DRIVER_PYTHON_OPTS='notebook'"
alias pyspark_use_shell="unset PYSPARK_DRIVER_PYTHON; unset PYSPARK_DRIVER_PYTHON_OPTS"

# Spark streaming pubsub connector
alias pyspark_pubsub="pyspark --jars ${SPARK_PUBSUB_JAR} --driver-class-path ${SPARK_PUBSUB_JAR} --py-files ${SPARK_PUBSUB_PYTHON_EGG}"
alias spark_submit_pubsub="spark-submit --jars ${SPARK_PUBSUB_JAR} --driver-class-path ${SPARK_PUBSUB_JAR} --py-files ${SPARK_PUBSUB_PYTHON_EGG}"
# alias spyder_spark_pubsub='SPYDER_PATH="$(which spyder)"; SPYDER_PY_PATH="$SPYDER_PATH.py"; cp $SPYDER_PATH $SPYDER_PY_PATH; spark-submit $SPYDER_PY_PATH --jars ${SPARK_PUBSUB_JAR} --driver-class-path ${SPARK_PUBSUB_JAR} --py-files ${SPARK_PUBSUB_PYTHON_EGG}'
