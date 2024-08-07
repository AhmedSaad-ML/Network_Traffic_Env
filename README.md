# Project Title
Reinforcement Learning for Intrusion Detection

## Description
- "Network_Traffic_Env" is a custom-built OpenAI Gym Environment. 
- This environment provides network traffic for training and evaluating reinforcement learning agents.
- It was built to accommodate any Preprocessed network traffic data and can be used for a variety of RL-based network traffic use cases.

## Installation
To use the environment you will need the following libraries and it was tested using those versions:
- python == 3.7.4
- gym == 0.18.0
- pandas == 0.25.1
- numpy == 1.17.2
- torch == 1.9.1


## Usage
- Make sure you have the libraries mentioned above installed.
- Example of how to use the environment is provided at the bottom of the "NetworkTrafficEnv.py" file (copy it to your agent's code).
- To use the environment, please clone the repo or download the "NetworkTrafficEnv.py" file and put it in the same directory as your RL Agent's code file.
- Then, import it in your agent's code as
  ```python
   from NetworkTrafficEnv import NetworkTrafficEnv
  

## Citation
If you use this code, please cite our [**paper**][doi]:

```bibtex
@inproceedings{saad2022reinforcement,
  address = {Cham},
  author = {Saad, Ahmed Mohamed Saad Emam and Yildiz, Beytullah},
  booktitle = {Computational Intelligence, Data Analytics and Applications},
  editor = {Garc{\'\i}a M{\'a}rquez, Fausto Pedro and Jamil, Akhtar and Eken, S{\"u}leyman and Hameed, Alaa Ali},
  isbn = {978-3-031-27099-4},
  pages = {230--243},
  publisher = {Springer International Publishing},
  title = {Reinforcement Learning for Intrusion Detection},
  year = {2022},
  url = {https://link.springer.com/chapter/10.1007/978-3-031-27099-4_18}
}
```

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

[doi]: https://link.springer.com/chapter/10.1007/978-3-031-27099-4_18
