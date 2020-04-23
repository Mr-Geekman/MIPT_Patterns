# Проект по курсу "Паттерны проектирования"

## Задание

* Выбрать open-source-проект. Например, на https://opensource.google/projects/explore/featured.
* Изучить его архитектуру, найти интересный модуль.
* Изобразить модуль в виде UML-диаграммы классов.
* Выделить паттерны и антипаттерны
* Для паттернов -- объяснить, какую конкретно задачу (проблему) они решают.
* Для антипаттернов -- предложить изменения в архитектуре, объяснить почему они улучшат проект.

## Отчет

В качестве проекта был выбран [AdaNet](https://github.com/tensorflow/adanet), а если быть конкретным, то два модуля: `subnetwork`, `ensemble`.

Начнем с того, что проанализируем все файлы и найдем классы и их описания.

### subnetwork

#### report.py

* Report -- A container for data to be collected about a :class:`Subnetwork`.

* MaterializedReport -- Data collected about a :class:`adanet.subnetwork.Subnetwork`.

#### generator.py

* TrainOpSpec -- A data structure for specifying training operations.

* Subnetwork -- An AdaNet subnetwork. A collection of weighted subnetworks form an AdaNet ensemble.

* Builder -- Interface for a subnetwork builder.

    * build_subnetwork_train_op -- Returns an op for training a new subnetwork.

    * build_subnetwork_report -- Returns a `subnetwork.Report` to materialize and record.

* Generator -- Interface for a candidate subnetwork generator.

    * generate_candidates -- Generates :class:`adanet.subnetwork.Builder` instances for an iteration.

* SimpleGenerator(Generator) -- Always generates the given :class:`adanet.subnetwork.Builder` instances.

### ensemble

#### ensembler.py

* TrainOpSpec -- A data structure for specifying ensembler training operations.

* Ensemble -- An abstract ensemble of subnetworks.

    * subnetworks -- Returns an ordered :class:`Iterable` of the ensemble's subnetworks.

* Ensembler -- An abstract ensembler.

    * build_ensemble -- Builds an ensemble of subnetworks. (returns Ensemble)

    * build_train_op -- Returns an op for training an ensemble.

#### strategy.py

* Candidate -- An ensemble candidate found during the search phase.
    
  Args:
    name: String name of this ensemble candidate.
    subnetwork_builders: Candidate :class:`adanet.subnetwork.Builder` instances to include in the ensemble.
    previous_ensemble_subnetwork_builders: :class:`adanet.subnetwork.Builder` instances to include from the previous ensemble.

* Strategy -- An abstract ensemble strategy.

    * generate_ensemble_candidates -- Generates ensemble candidates to search over this iteration.

    Args:
      subnetwork_builders: Candidate :class:`adanet.subnetwork.Builder` instances for this iteration.
      previous_ensemble_subnetwork_builders: :class:`adanet.subnetwork.Builder` instances from the previous ensemble. Including only a subset of these in a returned :class:`adanet.ensemble.Candidate` is equivalent to
        pruning the previous ensemble.
    Returns:
      An iterable of :class:`adanet.ensemble.Candidate` instances to train and consider this iteration.

* SoloStrategy(Strategy) -- Produces a model composed of a single subnetwork.

* GrowStrategy(Strategy) -- Greedily grows an ensemble, one subnetwork at a time.

* AllStrategy(Strategy) -- Ensembles all subnetworks from the current iteration.


#### mean.py

* MeanEnsemble(Ensemble) -- Mean ensemble.

* MeanEnsembler(Ensembler) -- Ensembler that takes the mean of logits returned by its subnetworks.

#### weighted.py

* WeightedSubnetwork -- An AdaNet weighted subnetwork.

  A weighted subnetwork is a weight applied to a subnetwork's last layer or logits (depending on the mixture weights type).
  Args:
    name: String name of :code:`subnetwork` as defined by its :class:`adanet.subnetwork.Builder`.
    iteration_number: 
    weight: 
    logits:
    subnetwork: The :class:`adanet.subnetwork.Subnetwork` to weight.

* ComplexityRegularized(Ensemble) -- An AdaNet ensemble where subnetworks are regularized by model complexity.

  Args:
    weighted_subnetworks: List of :class:`adanet.ensemble.WeightedSubnetwork` instances that form this ensemble. 
    bias: 
    logits: 
    subnetworks: List of :class:`adanet.subnetwork.Subnetwork` instances that form this ensemble. This is kept together with weighted_subnetworks for legacy reasons.
    complexity_regularization: Regularization to be added in the Adanet loss.

  Returns:
    An :class:`adanet.ensemble.Weighted` instance.

* MixtureWeightType -- Mixture weight types available for learning subnetwork contributions.

* ComplexityRegularizedEnsembler(Ensembler) -- The AdaNet algorithm implemented as an :class:`adanet.ensemble.Ensembler`.
    
    *  \_build_weighted_subnetwork -- Builds an `adanet.ensemble.WeightedSubnetwork`.


### Диаграмма классов

Исходники и картинки находятся в figures. При создании был использован [GenMyModel](https://app.genmymodel.com).


### Найденные паттерны
1. Команда:
    * subnetwork/generator/TrainOpSpec -- хранит операции для обучения. Позволяет гибко настраивать при каких конкретно опциях была получена оптимальная модель. Также позволяет разделить непосредственно использование сети и ее обучение. Когда мы используем сеть нам не нужно ничего знать про то, как она обучается.
    * ensemble/strategy/Candidate -- хранит текущих и предыдущих билдеров. При поиске оптимальной композиции нам может поднадобиться включать не только отдельные подсети, но и группы подсетей. Подсети же мы здесь храним не в виде самих подсетей, а в виде их строителей.

2. Фабричный метод:
    * subnetwork/generator/Builder -- абстракция для создания объектов с одинаковым интерфейсом. В алгоритме AdaNet производится перебор различных подсетей, чтобы построить некую хорошую композицию. Методов построения новых подсетей может быть много, но им нужен общий интерфейс, поэтому логично создать фабричный метод.
    * ensemble/ensemble/Ensembler -- абстранкция для создания объектов с одинаковым интерфейсом. При использовании различных алгоритмов ансамблирования надо чтобы у них был один и тот же интерфейс.

3. Строитель:
    * subnetwork/generator/Builder -- вынесение сложного процесса создания в отдельный класс. Реализация `Builder` должна уметь на основе требований к сети предложить какую-то новую подсеть с опциями для ее обучения. Так как мы разделяем сеть и ее обучение, то вполне разумно вынести содание комплекса из сети, алгоритма ее обучения в отдельный класс.
    * ensemble/ensemble/Ensembler -- вынесение сложного процесса создания в отдельный класс. Как и в с случае `Builder` нам в самом ансамбле не нужна сложная логика его создания, выбор алгоритмов для обучения.

4. Компоновщик:
    * subnetwork/generator/Subnetwork -- написано, что она хранит вдругие взвешенные подсети, т.е. по сути структура похожа на дерево. Позволяет удобно хранить композицию сетей.

5. Стратегия:
    * ensemble/strategy/Strategy -- семейство алгоритмов (наследники), которые можно использовать в коде взаимозаменяемо. Позволяет при вызове алгоритма перебора моделей просто использовать объект с одним и тем же интерфейсом, не специализируясь на конкретном принципе работы.


### Антипаттерны и ошибки

1. Дублирование класса `TrainOpSpec`. (Его внутренность идентична в обоих подмодулях.)
2. Использование внутри `ComplexityRegularized` и subnetworks и weighted_subnetworks (написано, что это legacy).
