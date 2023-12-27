dirname=$(dirname $0)
cd $dirname

if [ $# -ne 1 ]; then
    echo "错误：脚本需要一个参数 <job_name>"
    exit 1
fi

now=$(date "+%Y-%m-%d %H:%M:%S")
if ! [ -n "$(squeue -u $USER --long| grep $1)" ]; then 
    echo "$now $1 is not submited, submitting..."
    sbatch $1.job
    sleep 5
else
    echo "$now $1 is already submited, monitoring..."
fi

while true
do
    now=$(date "+%Y-%m-%d %H:%M:%S")
    if ! [ -n "$(squeue -u $USER --long| grep $1)" ]; then 
        echo "$now $1 is not submited, submitting..."
        sbatch $1.job
        sleep 5
    fi
    sleep 180
done