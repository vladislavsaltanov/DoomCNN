# DoomCNN: агент для ViZDoom deathmatch по пикселям

Нейросеть играет deathmatch на карте `cig.wad` MAP02 против 7 встроенных ботов ViZDoom [1, 25]. Она видит только экран
(4 последних серых кадра 320×240) и строку состояния: здоровье, броню, патроны, оружие. Каждый шаг сеть выбирает
четыре действия сразу: движение, поворот на точный угол, выстрел и смену оружия. Память — слой GRU [7]. Постановка взята из соревнований ViZDoom [2], где играли Arnold [3] и IntelAct [4]; рекуррентную сеть для частично наблюдаемой игры предлагали в DRQN [5].

Сеть обучена имитацией учителя-читера (behavior cloning [8] и DAgger [9]), а затем дообучена с подкреплением (PPO [13]). На
финальном test на базовой графике она набирает больше фрагов, чем учитель (15.6 против 11.9, 131 %); на незнакомой
графике — 41 % учителя (лучшая модель — 52 %).

Это вторая часть курсовой. Первая часть, агент для `defend_the_center`, лежит в ветке
[`doom/defend_the_center`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/defend_the_center) и кратко описана в конце.

![PPO: кривые обучения](reports/dm/e6_ppo/e6_ppo_kl.png)

## Финальный test (E7)

40 матчей на базовой графике и по 8 на 6 test-вариантах графики; ни эти seed, ни эти варианты не использовались ни
в обучении, ни при выборе моделей. Отчёт: [`reports/dm/e7_test/e7_test.md`](reports/dm/e7_test/e7_test.md), галерея с видео, нейронами и картами: [`e7_gallery.md`](reports/dm/e7_test/e7_gallery.md).

| критерий спека | значение | выполнен |
|---|---|---|
| база: ≥ 70 % фрагов учителя | **131 %** (15.57 против 11.88) | да |
| незнакомая графика: ≥ 60 % учителя | 41 % (лучшая модель E6d — 52 %) | **нет** |
| незнакомая графика: значимо больше ученика только на базовой графике | +3.81 [+2.83, +4.77] | да |
| память: M1 − M0 при равных данных, точность ответа | +0.64 [+0.03, +1.20] | да |

У итоговой модели память решает многое: с памятью GRU, обнуляемой на каждом шаге, она теряет 6.3 фрага из 9.7.
Ученики проваливаются на смешанных test-вариантах (новые текстуры и новый спрайт бота вместе): 0.5–2.3 фрага против
10–12 у учителя.

![test](reports/dm/e7_test/e7_ladder.png)

## Свежие матчи (выбор моделей)

Свежие матчи: 60 матчей по 2 минуты (40 на базовой графике и 20 на незнакомых вариантах), не использованные ни в
обучении, ни при выборе модели. Фраги за матч, в скобках 95-процентный bootstrap-интервал [22].

| модель | метод | фраги | доля учителя | первое место из 8 |
|---|---|---|---|---|
| учитель | скрипт-читер: A* [21], мгновенный прицел, видит только видимых ботов | 11.65 [10.58, 12.73] | 100 % | 68 % |
| **итоговая** (`ppo_e6c_01001k`) | + PPO 1 млн шагов со случайными точками появления и наградой за редкие районы | **13.20** [11.62, 14.75] | **113 %** | 73 % |
| `ppo_explore_01001k` (E6b) | + PPO 1 млн шагов с наградой за движение по карте | 11.77 [10.40, 13.08] | 101 % | 75 % |
| `ppo_kl_03000k` (E6) | DAgger-ученик + PPO 3 млн шагов | 10.22 [8.88, 11.53] | 88 % | 68 % |
| `dagger_sp_r2` | имитация + DAgger + пространственные головы | 4.83 [3.93, 5.68] | 41 % | 25 % |
| `dm3_m1_aux` | имитация (BC) + вспомогательные головы | 4.53 [3.73, 5.35] | 39 % | 17 % |

- Итоговая модель выше учителя: разность +1.55 [−0.25, +3.30]. На базовой графике 15.60 против 12.78.
- На незнакомой графике она набирает 89 % учителя (8.4 против 9.4).
- PPO прибавил к DAgger-ученику **+6.9 фрага [+5.8, +8.2]**: +5.4 основной прогон (E6) и ещё +1.55 [+0.48, +2.62]
  дообучение со штрафами за «пятачок» и выстрелы в пустоту и наградой за новые клетки (E6b).
- Финальная проверка на test (E7) впереди. Полные отчёты по срезам — [`reports/dm/README.md`](reports/dm/README.md).

Отрицательные и спорные результаты:

- **Имитация упёрлась примерно в 40 % учителя.** Ни больше данных, ни вспомогательные и пространственные головы,
  ни DAgger не сдвинули фраги за пределы шума. Причина — скрытая цель: учитель выбирает путь по своему маршруту и
  помнит подобранное, а по картинке этого не видно. В одинаковых кадрах ученик получает противоположные метки и учит
  их среднее [11, 12].
- **Подсказки учителя в PPO (kickstarting [16]) ломают политику.** С весом 1 за 250 тыс. шагов val упал с 3.57 до 1.06,
  застревание выросло до 42 %. Работает якорь к стартовой политике (KL).
- **Большой вес данных DAgger (×5) сломал движение:** застревание выросло до 15 %.
- **Выбор модели по 20–40 матчам шумный.** Лучшая на одних матчах может оказаться последней на других. Все выводы
  делаются только по свежим матчам с парными разностями.

## Как получена итоговая модель

1. **Учитель-читер** (`doomagent.dm.teacher`) по схеме привилегированного эксперта [10].
   - Знает геометрию карты (A* [21] по секторам) и места оружия.
   - Видит ботов только тех, что видны на экране (не меньше 50 пикселей): «читер по реакции, а не по знанию».
   - Режимы: бой, погоня, оружие, предметы, патруль.
   - Целится на точный угол за один шаг, уклоняется от снарядов, выбирает оружие по дистанции.
   - Улучшался циклом правок: каждая правка принималась только по приросту на свежих матчах.
2. **Данные:** R0 — 300 матчей на базовой графике, R1 — 192 матча на 16 вариантах графики (текстуры и спрайты),
   всего около 516 тыс. шагов.
3. **Имитация (BC)** [8]:
   - NatureCNN [6] и GRU [7] 512, четыре головы действий;
   - отрезки по 32 шага с прогревом памяти;
   - вспомогательные головы: бот в кадре, его столбец, предметы, клетка карты, направление, глубина [17];
   - отражение кадра и аугментации;
   - свёртки инициализированы итоговой моделью первой части.
4. **DAgger** [9]. Ученик играет, учитель размечает его шаги. Лучшая ветка — с пространственными головами: карта
   проходимости вокруг, одометрия, цель учителя.
5. **PPO** [13] (`doomagent.dm.ppo`, своя рекуррентная реализация на той же сети):
   - **асимметричный критик** [15]: он видит привилегированный вектор (время, счёт, боты за стенами), а политика — только
     экран;
   - **якорь KL** к стартовой политике;
   - вспомогательные потери продолжаются;
   - **формирующая награда** [19]: смерть −0.2, урон +0.005, предметы, оружие, застревание −0.01;
   - выбор действия с температурой 0.5;
   - 8 сред, 3 млн шагов, около 5 часов на RTX 3060 Ti;
   - дообучение E6b, ещё 1 млн шагов: 3/4 матчей на вариантах графики, награда за новые клетки карты [18], штрафы за
     «пятачок» (10 с в радиусе 192 единиц) и за выстрел при пустом экране;
   - E6c, ещё 1 млн шагов: в половине жизней бот при обучении появляется в случайной точке карты, награда за вход в
     редкие районы [18] (оценка всегда с обычным появлением).

Журнал всех решений, замеров и сбоев: [`research_log_dm.md`](research_log_dm.md).

## Структура (deathmatch)

| путь | что там |
| --- | --- |
| `src/doomagent/dm/` | среда, учитель, навигация, запись, модель, обучение (`train`, `dagger`, `ppo`, `rlenv`), оценка, отчёты, панель |
| `scripts/dm/` | этапы, отчёты, игра и видео (`dm_play.py`, `dm_demo_video.py`), PPO (`dm_ppo.py`), сравнения |
| `tests/dm/` | тесты (pytest; помеченные `vizdoom` требуют рабочий ViZDoom) |
| `reports/dm/` | отчёты по срезам, по папке на срез, с оглавлением |
| `research_log_dm.md` | журнал исследования deathmatch |

Модели (`runs/`), данные (`data/`) и видео в репозиторий не входят: они большие и воспроизводятся скриптами.

## Запуск

Установка — как в первой части, Windows: [`WINDOWS.md`](WINDOWS.md).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q tests/dm
```

Посмотреть игру модели или записать видео (35 кадров в секунду; `--boxes`, `--hud`, `--map` рисуются только для зрителя):

```powershell
.venv\Scripts\python scripts\dm\dm_play.py --who runs\dm\e6\ppo_e6c_01001k.pt --seed 36005 --boxes --hud --map
.venv\Scripts\python scripts\dm\dm_play.py --who runs\dm\e6\ppo_e6c_01001k.pt --seed 36005 --map --no-window --record artifacts\dm\e6_ppo\demo.mp4
```

Панель: живая игра, вид сети, кривые обучения — `.venv\Scripts\python -m doomagent.dm.dashboard`, затем
http://127.0.0.1:8766.

PPO от DAgger-ученика: `.venv\Scripts\python scripts\dm\dm_ppo.py --init runs\dm\dagger_sp_r2.pt --out runs\dm\ppo_kl --temp 0.5`.

## Первая часть: defend_the_center

Ветка [`doom/defend_the_center`](https://github.com/vladislavsaltanov/DoomCNN/tree/doom/defend_the_center), журнал [`research_log.md`](research_log.md).
Код первой части лежит и здесь (`src/doomagent/`, `scripts/`, `tests/`) без изменений.

Свёрточная сеть играет `defend_the_center` по 4 кадрам 160×120. Конвейер тот же: учитель → имитация → варианты
графики → DAgger → PPO со штрафом за кручение. На test итоговая модель наравне с учителем и на базовой графике
(22.97 против 22.89), и на незнакомой (21.15 против 20.96). Отсюда в deathmatch перешли конвейер, проверка на
свежих seed с парными разностями и свёртки итоговой модели как инициализация.

## Технологии

Сеть, обучение с учителем и анализ нейронов написаны на PyTorch [28], расчёты — на NumPy [31], кадры обрабатывает OpenCV [34], рисунки и графики строят matplotlib [33] и Pillow [35]; карты нейронов проецируются t-SNE [24] из scikit-learn [32]. Оптимизатор — Adam [20]; преимущества в PPO считаются по GAE [14]. Среда — ViZDoom [25] с графикой Freedoom [26]; варианты графики собираются правкой WAD-файлов [27]. Seed игры задаёт генератор Mersenne Twister [23]. Первая часть использует Stable-Baselines3 [29] и Gymnasium [30]. Тесты — pytest [36], видео склеивает FFmpeg [37].

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
