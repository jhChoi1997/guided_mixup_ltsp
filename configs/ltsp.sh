python run.py --gpus 6 -c ./configs/ltsp/ToyCar.yaml &
python run.py --gpus 6 -c ./configs/ltsp/ToyConveyor.yaml &
python run.py --gpus 6 -c ./configs/ltsp/fan.yaml &
python run.py --gpus 7 -c ./configs/ltsp/pump.yaml &
python run.py --gpus 7 -c ./configs/ltsp/slider.yaml &
python run.py --gpus 7 -c ./configs/ltsp/valve.yaml &

wait 
