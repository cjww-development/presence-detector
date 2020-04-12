# presence-detector

## How to run
All environment dependencies are included in the `requirements.txt` file.

To generate a conda environment from this file, run
```shell script
conda create --name <env_name> --file requirements.txt
```

To activate the env
```shell script
conda activate <env_name>
```

To get the learning model
```shell script
./get-model
```

To run the project
```shell script
python3 -m presencedetector
```

## Modes
Presence detector currently runs in two modes as defined by `mode.static` in `config/application.conf`.

##### Static Mode
Takes an input image defined by conf value `obj-detector.input-path`. To get started this an image of a street in london.
The inference result will get dumped into file `output.jpg`, defined by conf value `obj-detector.output-path.static`

##### Video Mode
Takes video input from the first available system camera. The inference result is dumped into the path defined by `obj-detector.output-path.video`. The result is an AVI file.

### License
This code is open source software licensed under the [Apache 2.0 License]("http://www.apache.org/licenses/LICENSE-2.0.html")
