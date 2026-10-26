# DoomCNN: агенты для ViZDoom по пикселям

Курсовая работа о том, как научить нейросеть играть в Doom (ViZDoom), видя только экран. Работа состоит из двух
частей. В обеих используется один конвейер:

```
учитель-скрипт с привилегированными данными → имитация (behavior cloning) → варианты графики → DAgger → PPO
```

Каждая часть лежит в своей ветке с подробным README, журналом исследования, отчётами и тестами.

| часть | задача | итог | ветка |
|---|---|---|---|
| **1. defend_the_center** | стоять в центре арены и отстреливать подходящих монстров | **наравне с учителем** и на базовой графике, и на незнакомой | [`doom/defend_the_center`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/defend_the_center) |
| **2. deathmatch MAP02** | deathmatch против 7 ботов: ходить, собирать оружие, выбирать его, стрелять; сеть с памятью | на test **сильнее учителя-читера на базовой графике** (131 % его фрагов), на незнакомой графике — 41 % | [`doom/deathmatch`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/deathmatch) |

## Часть 1. defend_the_center

Свёрточная сеть видит 4 последних серых кадра 160×120 и выбирает одно из 5 действий: два размера поворота в каждую
сторону и выстрел. Итог на test (100 эпизодов на базе, 11 test-вариантов графики по 20 эпизодов), средняя награда
за эпизод:

| модель | базовая графика | незнакомая графика |
|---|---|---|
| учитель (скрипт по labels buffer) | 22.89 [22.61, 23.14] | 20.96 [20.42, 21.45] |
| **итоговая** (имитация + варианты + DAgger + PPO со штрафом за кручение) | **22.97** [22.75, 23.19] | **21.15** [20.50, 21.72] |
| только имитация на базовой графике | 23.35 [22.63, 24.02] | 3.96 [3.41, 4.56] |
| PPO с нуля, равный бюджет | 16.66 [16.17, 17.18] | 12.50 [11.87, 13.09] |

Все критерии спецификации выполнены. Главный урок: на незнакомой графике решили варианты графики, DAgger и PPO
(+17 к модели, обученной только на базовой графике).

![defend_the_center: незнакомая графика](reports/s8_ood.png)

Подробно: ветка [`doom/defend_the_center`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/defend_the_center),
журнал [`research_log.md`](research_log.md), отчёт [`reports/s8_final.md`](reports/s8_final.md).

## Часть 2. deathmatch MAP02

Сеть видит 4 серых кадра 320×240 и строку состояния. Каждый шаг она выбирает четыре действия сразу: движение,
поворот на точный угол, выстрел и смену оружия. Память — слой GRU.

Учитель — скрипт-«читер по реакции»: знает карту и места оружия, целится мгновенно, но ботов видит только на экране.

Итог на 60 свежих матчах по 2 минуты:

| модель | фраги за матч | доля учителя | первое место из 8 |
|---|---|---|---|
| учитель | 11.65 [10.58, 12.73] | 100 % | 68 % |
| **итоговая** (DAgger-ученик + PPO 5 млн шагов: награда за движение по карте, случайные точки появления при обучении) | **13.20** [11.62, 14.75] | **113 %** | 73 % |
| после основного PPO (3 млн шагов) | 10.22 [8.88, 11.53] | 88 % | 68 % |
| имитация + DAgger | 4.83 [3.93, 5.68] | 41 % | 25 % |
| только имитация | 4.53 [3.73, 5.35] | 39 % | 17 % |

- На базовой графике итоговая модель выше учителя: 15.60 против 12.78.
- На незнакомой графике она набирает 89 % учителя.
- Имитация упёрлась в 40 %: учитель решает по своей скрытой цели, а её не видно на картинке. Дальше вывел PPO
  со своей рекуррентной реализацией, асимметричным критиком и якорем к стартовой политике.
- Финальный test (40 матчей на базе, 48 на test-вариантах графики): база 15.57 против 11.88 у учителя (131 %), незнакомая
  графика 41 % учителя (лучшая модель 52 %); с памятью, обнуляемой на каждом шаге, модель теряет 6.3 фрага из 9.7.

![deathmatch: кривые PPO](https://github.com/vladislavsaltanov/DoomCNN/blob/doom/deathmatch/reports/dm/e6_ppo/e6_ppo_kl.png?raw=true)

Подробно: ветка [`doom/deathmatch`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/deathmatch) — README,
журнал `research_log_dm.md`, отчёты по срезам `reports/dm/` с оглавлением.

## Как открыть ветку

```powershell
git clone https://github.com/vladislavsaltanov/DoomCNN.git
cd DoomCNN
git checkout doom/deathmatch          # или doom/defend_the_center
```

Установка и запуск описаны в README каждой ветки. Для Windows — в [`WINDOWS.md`](WINDOWS.md).

## Ветки

| ветка | что там |
|---|---|
| `main` | этот обзор и код первой части |
| `doom/defend_the_center` | первая часть целиком: код, отчёты, журнал, итоговый README |
| `doom/deathmatch` | вторая часть: код первой части без изменений и deathmatch (`doomagent.dm`), отчёты, журнал |

## Список литературы

Публикации:

1. Kempka M. и др. ViZDoom: A Doom-based AI research platform for visual reinforcement learning // 2016 IEEE Conference on Computational Intelligence and Games (CIG). 2016. P. 1–8. arXiv:1605.02097.
2. Wydmuch M., Kempka M., Jaśkowski W. ViZDoom Competitions: Playing Doom from Pixels // IEEE Transactions on Games. 2019. Vol. 11, № 3. P. 248–259. arXiv:1809.03470.
3. Lample G., Chaplot D. S. Playing FPS Games with Deep Reinforcement Learning // Proceedings of the AAAI Conference on Artificial Intelligence. 2017. Vol. 31, № 1. arXiv:1609.05521.
4. Dosovitskiy A., Koltun V. Learning to Act by Predicting the Future // International Conference on Learning Representations (ICLR). 2017. arXiv:1611.01779.
5. Hausknecht M., Stone P. Deep Recurrent Q-Learning for Partially Observable MDPs // AAAI Fall Symposium Series. 2015. arXiv:1507.06527.
6. Mnih V. и др. Human-level control through deep reinforcement learning // Nature. 2015. Vol. 518. P. 529–533.
7. Cho K. и др. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation // Proceedings of EMNLP. 2014. P. 1724–1734. arXiv:1406.1078.
8. Pomerleau D. A. ALVINN: An Autonomous Land Vehicle in a Neural Network // Advances in Neural Information Processing Systems 1. 1989. P. 305–313.
9. Ross S., Gordon G., Bagnell D. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning // Proceedings of AISTATS, PMLR. 2011. Vol. 15. P. 627–635. arXiv:1011.0686.
10. Chen D., Zhou B., Koltun V., Krähenbühl P. Learning by Cheating // Proceedings of the Conference on Robot Learning (CoRL), PMLR. 2019. Vol. 100. P. 66–75. arXiv:1912.12294.
11. Codevilla F. и др. End-to-end Driving via Conditional Imitation Learning // 2018 IEEE International Conference on Robotics and Automation (ICRA). 2018. P. 4693–4700. arXiv:1710.02410.
12. Weihs L. и др. Bridging the Imitation Gap by Adaptive Insubordination // Advances in Neural Information Processing Systems 34. 2021. arXiv:2007.12173.
13. Schulman J. и др. Proximal Policy Optimization Algorithms. 2017. arXiv:1707.06347.
14. Schulman J. и др. High-Dimensional Continuous Control Using Generalized Advantage Estimation // International Conference on Learning Representations (ICLR). 2016. arXiv:1506.02438.
15. Pinto L. и др. Asymmetric Actor Critic for Image-Based Robot Learning // Robotics: Science and Systems. 2018. arXiv:1710.06542.
16. Schmitt S. и др. Kickstarting Deep Reinforcement Learning. 2018. arXiv:1803.03835.
17. Jaderberg M. и др. Reinforcement Learning with Unsupervised Auxiliary Tasks // International Conference on Learning Representations (ICLR). 2017. arXiv:1611.05397.
18. Bellemare M. и др. Unifying Count-Based Exploration and Intrinsic Motivation // Advances in Neural Information Processing Systems 29. 2016. P. 1471–1479. arXiv:1606.01868.
19. Ng A. Y., Harada D., Russell S. Policy invariance under reward transformations: Theory and application to reward shaping // Proceedings of the 16th International Conference on Machine Learning (ICML). 1999. P. 278–287.
20. Kingma D. P., Ba J. Adam: A Method for Stochastic Optimization // International Conference on Learning Representations (ICLR). 2015. arXiv:1412.6980.
21. Hart P. E., Nilsson N. J., Raphael B. A Formal Basis for the Heuristic Determination of Minimum Cost Paths // IEEE Transactions on Systems Science and Cybernetics. 1968. Vol. 4, № 2. P. 100–107.
22. Efron B., Tibshirani R. J. An Introduction to the Bootstrap. New York: Chapman & Hall, 1993. 436 p.
23. Matsumoto M., Nishimura T. Mersenne Twister: a 623-dimensionally equidistributed uniform pseudorandom number generator // ACM Transactions on Modeling and Computer Simulation. 1998. Vol. 8, № 1. P. 3–30.
24. van der Maaten L., Hinton G. Visualizing Data using t-SNE // Journal of Machine Learning Research. 2008. Vol. 9. P. 2579–2605.

Программные средства и данные:

25. ViZDoom 1.3.2 [Электронный ресурс]. URL: https://github.com/Farama-Foundation/ViZDoom
26. Freedoom: free content game based on the Doom engine [Электронный ресурс]. URL: https://freedoom.github.io/
27. Doom Wiki. WAD [Электронный ресурс]. URL: https://doomwiki.org/wiki/WAD
28. Paszke A. и др. PyTorch: An Imperative Style, High-Performance Deep Learning Library // Advances in Neural Information Processing Systems 32. 2019. arXiv:1912.01703. Версия 2.14.1.
29. Raffin A. и др. Stable-Baselines3: Reliable Reinforcement Learning Implementations // Journal of Machine Learning Research. 2021. Vol. 22, № 268. P. 1–8. Версия 2.9.0.
30. Towers M. и др. Gymnasium: A Standard Interface for Reinforcement Learning Environments. 2024. arXiv:2407.17032. Версия 1.4.0.
31. Harris C. R. и др. Array programming with NumPy // Nature. 2020. Vol. 585. P. 357–362. Версия 2.5.4.
32. Pedregosa F. и др. Scikit-learn: Machine Learning in Python // Journal of Machine Learning Research. 2011. Vol. 12. P. 2825–2830. Версия 1.9.1.
33. Hunter J. D. Matplotlib: A 2D Graphics Environment // Computing in Science & Engineering. 2007. Vol. 9, № 3. P. 90–95. Версия 3.11.2.
34. Bradski G. The OpenCV Library // Dr. Dobb's Journal of Software Tools. 2000. Версия opencv-python 5.0.0.93.
35. Pillow (PIL Fork) 13.0.0 [Электронный ресурс]. URL: https://python-pillow.org/
36. pytest 9.1.1 [Электронный ресурс]. URL: https://docs.pytest.org/
37. FFmpeg [Электронный ресурс]. URL: https://ffmpeg.org/

## Благодарности

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom), [Freedoom](https://freedoom.github.io/),
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), PyTorch.


Отдельное спасибо [Claude Code](https://claude.com/claude-code) (Anthropic, Opus 5.5).
