#! /bin/sh

# default values
input="INPUT_CHECKPOINT_NAME"
episodes=5
expname="DEF_EXP"
while [ $# -gt 0 ] ; do
  case $1 in
    -i | --input) input="$2" ;;
    -e | --episodes) episodes="$2" ;;
    -n | --name) expname="$2" ;;
  esac
  shift
done

echo "Extract Setup"
echo ">> CHECKPOINT NAME: $input"
echo ">> EPISODES: $episodes"
echo ">> EXPERIMENT NAME: $expname"

while true; do
    read -p "Start? (y/n)" yn
    case $yn in
        [Yy] ) break;;
        [Nn] ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

# change mount paths to your local paths
# replace 'HOST_PATH_CHANGE_THIS' with your path
docker run -d \
  --name scobots-extract-$expname \
  --mount type=bind,src=/HOST_PATH_CHANGE_THIS/baselines_focusfiles,dst=/workdir/extract/baselines/baselines_focusfiles \
  --mount type=bind,src=/HOST_PATH_CHANGE_THIS/baselines_extract_input,dst=/workdir/extract/baselines/baselines_extract_input \
  --mount type=bind,src=/HOST_PATH_CHANGE_THIS/baselines_extract_output,dst=/workdir/extract/baselines/baselines_extract_output \
	scobots-extract:latest \
	-i $input \
	-e $episodes \
  -n $expname