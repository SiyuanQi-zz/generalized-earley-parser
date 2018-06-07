# ICML2018 - Generalized Earley Parser: Bridging Symbolic Grammars and Sequence Data for Future Prediction

The algorithm is described in a [ICML 2018 paper](http://web.cs.ucla.edu/~syqi/publications/icml2018earley/icml2018earley.pdf) ([supplementary materials](http://web.cs.ucla.edu/~syqi/publications/icml2018earley/icml2018earley_supplementary.pdf)).

The **generalized Earley parser** is implemented in `src/python/parser/generalizedearley.py`.

Folders in the repository:
- `src/python/datasets`: code for loading the extracted features of the used datasets.
- `src/python/experiments`: code for running the experiments. The main experiments are implemented in `cad.py` and `wnp.py`.
- `src/python/parser`: files for grammar inference.


***
If you find this code useful, please cite our work with the following bibtex:
```
@inproceedings{qi2018future,
    title={Generalized Earley Parser: Bridging Symbolic Grammars and Sequence Data for Future Prediction},
    author={Qi, Siyuan and Jia, Baoxiong and Zhu, Song-Chun},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2018}
}
```
