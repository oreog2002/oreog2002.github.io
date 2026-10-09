---
layout: post
title: "NFL QB Performance Predictor" 
image: "/posts/nfl-title-img.png"
tags: [NFL, RNN, Python, PyTorch, Scikit-learn, Matplotlib]
editor_options: 
  markdown: 
    wrap: 72
---

# Table of Contents

- [00. Abstract](#abstract)
- [01. Introduction](#intro)
  - [Motivation](#intro-motivation)
  - [Results](#intro-results)
- [02. Background Information](#background-info)
- [03. Machine Learning Background](#ML-background)
  - [Model Architecture](#ML-background-model)
- [04. Data and Analysis](#data-analysis)
  - [Example Walk Through of Both Models](#example)
  - [Improvements](#improvements)
  - [Limitations](#limitations)
  - [What We Learned](#learnings)
- [05. Conclusion](#conclusion)
- [06. References](#references)

------------------------------------------------------------------------

# Abstract <a name="abstract"></a>

Predicting player performance in NFL games has become a critical tool
for fantasy football managers, sports analysts, and bettors. Using data
from nfl-data-py, this project explores the potential of a neural
network regression model to forecast weekly player performance metrics
such as rushing yards, passing yards, receiving yards, and touchdowns.
These predictions are tailored for practical applications in fantasy
football and sports betting. Using extensive play-by-play data and
incorporating defensive metrics, our approach improves on traditional
models, which focus primarily on offensive statistics. The results
demonstrate the model's ability to produce accurate, actionable
forecasts, providing insights into its significance for real-world
decision-making.

# Introduction <a name="intro"></a>

Fantasy football has become a massive cultural phenomenon, generating
\$30.5 billion in revenue in 2023 with projections to exceed \$114.7
billion by 2033 [1](#references). This growth is fueled by
an ever-increasing number of participants who rely on data-driven
insights to set optimal weekly lineups and gain a competitive edge.
Similarly, sports betting has exploded in popularity, with millions of
bettors seeking accurate predictions to evaluate player-based prop bets.
The NFL, as the most popular professional sports league in the United
States, has become a central focus for these industries. Thus,
predicting NFL player performance has evolved into a cornerstone of
modern sports analytics. Fantasy football participants and sports
bettors alike rely on predictions to decide who to start in weekly
lineups or determine whether players will exceed key metrics such as
passing yards, touchdowns, and interceptions.

To address the demand for accurate and actionable insights, this project
uses machine learning to predict quarterback (QB) performance for
hypothetical future games. Using data from the open source nfl-data-py
library, which offers detailed play-by-play data for every NFL game, we
developed a model that forecasts key metrics such as passing yards,
touchdowns, interceptions, and completion percentage. These metrics
align with fantasy football scoring categories and are critical for
evaluating sports betting opportunities. Our model is built on a
Long-Short-Term Memory (LSTM) neural network, a deep learning
architecture that excels at capturing sequential patterns and temporal
dependencies. By integrating modern machine learning techniques,
including embeddings, attention mechanisms, and sequence normalization,
the model delivers predictions that reflect the unpredictable and
dynamic nature of NFL games.

## Motivation <a name="intro-motivation"></a>

The motivation behind this project stems from the intersection of two
rapidly growing industries: fantasy football and sports betting.
Accurate player performance predictions are invaluable to fantasy
football participants, who can use these insights to make start/sit
decisions, navigate trades, and evaluate waiver wire pickups. For sports
bettors, player-based predictions help identify valuable betting
opportunities, such as yardage over/under, touchdown probabilities, or
other player-specific prop bets. With more than 29 million active
fantasy football participants in 2022 [3](#references) and a booming sports
betting market that saw \$16 billion bets on Super Bowl 57, the
potential for data-driven decision-making is immense.

The core of this project is driven by a desire to explore the intricate
relationships between game-related factors and player statistics.
Quarterback performance, in particular, is influenced by a variety of
contextual variables, including their recent form, the strength of
opposing defenses, and situational game factors such as weather or
venue. By analyzing detailed play-by-play data, this project seeks to
uncover performance patterns that might not be apparent through
traditional observation alone. The ultimate goal is to create a model
that bridges the gap between raw NFL data and actionable insights,
providing users with a competitive edge in their fantasy leagues or
betting endeavors.

The effectiveness of the model will be evaluated using real-time data
from the 2024 NFL season. Predictions will be compared against actual
game statistics, with metrics such as Mean Squared Error (MSE) used to
measure accuracy. By incorporating weekly feedback, the model will adapt
to evolving conditions throughout the season, ensuring that its
predictions remain relevant and reliable. This iterative evaluation
process not only highlights the robustness of the model but also
demonstrates its practical utility in real world scenarios.

## Results <a name="intro-results"></a>

The model predicts five key quarterback performance metrics: passing
yards, touchdowns, interceptions, completion percentage, and the number
of times the quarterback gets sacked. These predictions provide
actionable insights for a range of users, including fantasy football
participants and sports bettors. Fantasy football players can leverage
the model to identify quarterbacks poised to outperform or underperform
expectations, optimizing weekly lineup decisions. For sports bettors,
the model offers data-driven analysis for player prop bets, providing
insights into yardage over/unders, touchdown probabilities, and
interception risks.

The model's effectiveness will be assessed using real-time data from the
2024 NFL season. Predictions will be compared against actual game
outcomes, with accuracy evaluated using metrics such as Mean Squared
Error (MSE).

# Background Information <a name="background-info"></a>

The NFL remains the most popular professional sports league in the
United States, commanding immense fan engagement and driving significant
economic impact. In 2022, over 29 million people participated in fantasy
football, contributing to an \$11 billion industry [3](#references). By
2023, this figure had skyrocketed to over \$30 billion
[1](#references), underscoring the immense engagement
fantasy football generates for the NFL. Simultaneously, sports betting
has experienced exponential growth, fueled by state-by-state
legalization. For instance, betting on the Super Bowl jumped from \$7.6
billion in 2022 to \$16 billion in 2023 [2](#references).
These trends highlight the intertwined impacts of fantasy football and
sports betting on NFL engagement.

The rise of machine learning in sports analytics has further transformed
how games are analyzed and decisions are made. NFL broadcasts now
regularly feature machine learning models estimating probabilities for
pivotal moments, such as fourth-down conversions, offering fans a
data-driven lens on strategy. Our model applies similar techniques to
individual player performance, helping users decide whether a player
should be started in a fantasy lineup or if their projected statistics
align with favorable betting odds.

The intersection of player evaluation and sports gaming has been
explored in other contexts, such as Madden NFL video game ratings.
However, static ratings, like those used in Madden, fail to account for
real-time adaptability and nuanced changes throughout the season. This
project aims to overcome these limitations by leveraging dynamic,
context-sensitive machine learning tools that reflect the complexities
of NFL gameplay.

The competitive landscape of fantasy football and sports betting tools
has spurred significant innovation. Apps like WalterPicks compare their
projections against established platforms, such as ESPN, to provide
users with data-driven advice on start/sit decisions, trades, and waiver
pickups. This landscape underscores the need for continual improvement
in prediction accuracy and reliability. By combining extensive NFL
knowledge with machine learning, our project strives to provide
actionable insights that stand out in the evolving world of sports
analytics.

# Machine Learning Background <a name="ML-background"></a>

The development of our quarterback performance prediction system began
with a foundational Long Short-Term Memory (LSTM) neural network
architecture. This initial model was designed to process sequential
game-level data for both quarterbacks and opposing defenses. Separate
LSTM layers were employed to analyze quarterback and defensive
histories, incorporating up to 32 games of quarterback statistics and 16
games of defensive statistics. The sequences included limited features,
such as passing yards, touchdowns, air yards, sacks, and yards after the
catch. Static embeddings were used to numerically represent quarterback
and team identities, and the outputs of the LSTM layers were
concatenated with these embeddings. The combined feature vector was
passed through a series of fully connected layers to produce predictions
for five core game-level metrics: passing yards, touchdowns,
interceptions, completion percentage, and yards per attempt.

While the initial model provided a functional baseline for analyzing
quarterback and defensive trends, it exhibited several key shortcomings.
First, it treated all games within a sequence equally, failing to
emphasize the higher predictive value of recent performances. For
instance, a quarterback's performance in the last few games typically
provides stronger signals for future outcomes than games earlier in the
season. Without temporal weighting, the model struggled to account for
evolving trends, such as improvements in a quarterback's decision-making
or adjustments in a defense's strategy. Second, the feature set was
narrow, excluding critical contextual variables such as weather
conditions, home/away status, and detailed passing breakdowns by
direction or depth. These omissions limited the model's ability to
account for situational nuances, such as a defense's performance in
extreme weather or against mobile quarterbacks.

Additionally, the initial model lacked an attention mechanism, which
restricted its ability to identify and prioritize the most relevant
portions of historical sequences. This limitation meant that all games
contributed equally to predictions, even when some were less relevant
due to outlier performances or changing game contexts. The architecture
itself was relatively shallow, comprising only a few LSTM layers and
simple fully connected layers. This restricted the model's capacity to
learn complex interactions between features. Training techniques were
also basic, relying solely on the Adam optimizer with minimal
regularization. Consequently, the model often overfit the training data
and failed to generalize effectively to unseen scenarios.

Recognizing these limitations, we transitioned to a more advanced
architecture that incorporates state-of-the-art techniques in sequence
modeling. The enhanced model replaces the LSTM-based approach with a
transformer-based architecture, which excels at capturing long-range
dependencies in sequential data. Transformer layers utilize multi-head
self-attention mechanisms to dynamically identify and prioritize the
most informative parts of a sequence. For example, the model can focus
on a quarterback's recent success against strong pass rushes or a
defense's struggles with deep passing plays. By dynamically weighting
sequence elements, the transformer architecture ensures that predictions
are guided by the most relevant historical trends.

Another key improvement in the enhanced model is its ability to
incorporate temporal weighting directly into the sequence processing.
Historical data is weighted exponentially, giving greater importance to
recent games while preserving the context provided by earlier
performances. This approach reflects the intuition that a quarterback's
or defense's recent form is often a stronger predictor of future
outcomes than older data.

The enhanced model also benefits from a significantly expanded feature
set, including advanced metrics like sack rates, deep pass rates,
passing direction breakdowns, and yards per attempt. Additional
contextual features, such as weather conditions, home/away status, and
the quality of the receiving corps, are included to provide a more
comprehensive view of the factors influencing quarterback performance.
These features allow the model to account for situational variations
that were previously overlooked, such as the impact of weather on deep
passing accuracy or the role of a strong receiving corps in a
quarterback's efficiency.

In addition to the architectural advancements, the enhanced model
incorporates more sophisticated training and regularization techniques.
Batch normalization and layer normalization are applied to stabilize
learning by ensuring consistent input scales across layers. Dropout
regularization and gradient clipping are used to prevent overfitting and
mitigate exploding gradients, enhancing the model's robustness and
ability to generalize. Identity embeddings for quarterbacks and teams
have also been improved to capture latent attributes, such as a
quarterback's mobility, preferred passing depth, or a defense's blitz
frequency. These embeddings provide the model with a richer
understanding of individual entities and their unique characteristics.

## Model Architecture <a name="ML-background-model"></a>

<figure id="fig:model_architecture" data-latex-placement="H">
<img src="./archeture.png" style="width:100.0%" />
<figcaption>Model Architecture for QB Core Stats Prediction. The
architecture integrates QB game history, defensive history, and
embeddings for QB and team identities. Sequential data is processed
through LSTM layers, while embeddings are used for static
representations. The concatenated features are refined through dense
layers with dropout regularization, producing predictions for passing
yards, touchdowns, interceptions, completion percentage, and yards per
attempt.</figcaption>
</figure>

The initial model architecture, shown in Figure
[1](#fig:model_architecture){reference-type="ref"
reference="fig:model_architecture"}, was built on a foundation of Long
Short-Term Memory (LSTM) networks to capture sequential trends in
quarterback (QB) and defensive history. The architecture processed up to
16 games of QB statistics and 8 games of defensive statistics,
normalizing and passing them through two-layer LSTM stacks for temporal
feature extraction. These sequences were complemented by static
embeddings for QB and team identities, which provided numerical
representations of these categorical inputs. After concatenating the
LSTM outputs and embeddings, the model utilized a series of dense layers
with dropout regularization to make predictions for five core
quarterback performance metrics: passing yards, touchdowns,
interceptions, completion percentage, and yards per attempt.

While functional, this architecture had several limitations. The use of
LSTMs, although effective for sequence modeling, restricted the ability
to emphasize the most relevant parts of the sequence. All historical
games were treated equally, missing the higher predictive importance of
recent performances. Additionally, the reliance on shallow fully
connected layers and the absence of an attention mechanism limited the
model's ability to capture complex interactions between features,
especially in dynamic game environments. Despite its ability to
generalize to some extent, the architecture struggled with overfitting
and failed to fully utilize the rich context available in the data.

<figure id="fig:transformer_model_architecture"
data-latex-placement="H">
<img src="./transformer.png" style="width:80.0%" />
<figcaption>Enhanced Model Architecture: Transformer-Based Design for QB
Core Stats Prediction. The model integrates QB and defensive play
histories, embeddings for QB and team identities, and attention
mechanisms. Transformer layers extract rich patterns, with outputs
refined into predictions for key performance metrics.</figcaption>
</figure>

In contrast, the enhanced model, shown in Figure
[2](#fig:transformer_model_architecture){reference-type="ref"
reference="fig:transformer_model_architecture"}, represents a
significant evolution in the architecture by replacing LSTMs with
transformer layers for sequential data processing. Transformer layers,
renowned for their ability to handle long-range dependencies, are better
suited for capturing nuanced relationships in sequential data. By
introducing multi-head attention mechanisms, the model dynamically
identifies and emphasizes the most relevant portions of the QB and
defensive play sequences. This enables it to focus on critical plays or
trends, such as a QB's recent performance under pressure or a defense's
struggles against deep passes.

The enhanced model retains embeddings for QB and team identities but
integrates them more effectively with the transformer outputs. These
embeddings now interact with richer temporal features extracted by
multiple transformer layers. The model's prediction network is expanded
to include both the main performance metrics (e.g., passing yards,
touchdowns, and interceptions) and auxiliary metrics like touchdown rate
and interception rate. This additional granularity improves the
interpretability of the predictions.

Notably, the enhanced architecture incorporates advanced data
preparation techniques and sequence weighting. Recent games are given
greater importance through exponential weighting, ensuring that
predictions reflect current form while maintaining the context of
longer-term trends. Additionally, the prediction network benefits from
deeper fully connected layers, improved dropout regularization, and
advanced normalization techniques, which enhance its robustness and
generalization capabilities.

By transitioning from LSTM-based temporal modeling to a
transformer-based architecture, the enhanced model demonstrates
significant improvements in both predictive accuracy and flexibility. It
captures richer patterns in historical data, dynamically adapts to
contextual variations, and produces more precise predictions across
diverse game scenarios. This evolution underscores the effectiveness of
transformer architectures in tackling complex sequence prediction tasks
in sports analytics.

# Data and Analysis <a name="data-analysis"></a>

The performance of the original LSTM-based model during training is
illustrated in the chart titled QB Core Stats Prediction - Training
History. This chart tracks both training loss (blue line) and validation
loss (red line) over 40 epochs, using Mean Squared Error (MSE) as the
performance metric. The training loss begins at a high value of
approximately 1.3 MSE but steadily decreases as the model learns from
the data, eventually reaching a minimum of 0.8098 by the 40th epoch.
Similarly, the validation loss stabilizes around 1.0383, indicating
effective generalization to unseen data without significant overfitting.
These results highlight the effectiveness of the LSTM-based architecture
in processing sequential data and balancing the model's learning from
historical quarterback and defensive statistics.

Despite the solid performance of the LSTM-based model, certain trends in
the training history indicate areas for improvement. For instance, while
the training loss steadily decreases, the validation loss plateaus
relatively early, suggesting that the model has limited capacity to
further reduce error on unseen data. This limitation is likely a result
of the fixed sequence length and the inability of LSTM layers to
dynamically prioritize the most relevant portions of historical data.

<figure id="fig:training_history_loss" data-latex-placement="H">
<img src="./training_history_20241122_173158.png"
style="width:100.0%" />
<figcaption>Training and Validation Loss Over 40 Epochs for the QB Core
Stats Prediction Model. The training loss (blue) steadily decreases,
reaching a minimum of 0.8098, while the validation loss (red) stabilizes
at 1.0383, indicating effective learning and minimal
overfitting.</figcaption>
</figure>

In contrast, the transformer-based model demonstrates notable
advancements in both training dynamics and overall performance, as
illustrated in the new training history chart. The transformer model
introduces a more complex architecture, incorporating multi-head
attention mechanisms and the ability to process up to 2,000 plays per
quarterback or defensive sequence. The training loss (solid blue line)
and validation loss (solid red line) show clear improvement compared to
the earlier LSTM model, with the validation loss reaching a lower
minimum value of 0.8814.

Additionally, the chart tracks auxiliary losses for secondary metrics
(e.g., touchdown rate and interception rate) to ensure that the model
learns relationships across multiple performance dimensions. The
auxiliary training loss (dotted blue line) and auxiliary validation loss
(dotted red line) demonstrate consistent reductions, highlighting the
transformer model's ability to process richer datasets and predict a
wider range of metrics effectively.

<figure id="fig:new_training_history_loss" data-latex-placement="H">
<img src="./newtraininghistory.png" style="width:100.0%" />
<figcaption>Training and Validation Loss for the Transformer-Based
Model. The inclusion of auxiliary losses for secondary metrics
demonstrates the model’s capacity to learn multi-dimensional performance
relationships. Validation loss reaches a minimum of 0.8814, highlighting
improved generalization.</figcaption>
</figure>

The enhanced training dynamics of the transformer-based model can be
attributed to several key improvements. By processing a larger play
history of up to 2,000 plays and applying exponential weighting, the
model effectively prioritizes recent data without discarding the context
provided by earlier performances. Multi-head attention mechanisms allow
the transformer to dynamically focus on the most relevant parts of the
sequence, enabling it to uncover nuanced relationships between
quarterback tendencies, defensive schemes, and game contexts.

## Example Walk Through of Both Models <a name="example"></a>

To illustrate how the model processes data and generates predictions, we
present an example forecasting Patrick Mahomes' performance in a
hypothetical matchup against the Buffalo Bills. This walkthrough
demonstrates how raw data, such as quarterback and defensive statistics,
flows through the model to produce actionable predictions.

The process begins by retrieving input data for the quarterback and the
opposing defense. For Mahomes, the model processes statistics from his
last 16 games, such as 331 passing yards and 3 touchdowns in the most
recent game. For the Bills, the model processes defensive statistics
from their last 8 games, such as 245 passing yards allowed and 1
interception in their most recent game. This input data is visualized in
Figures [5](#fig:qb_history){reference-type="ref"
reference="fig:qb_history"} and
[6](#fig:defense_history){reference-type="ref"
reference="fig:defense_history"}.

<figure id="fig:qb_history" data-latex-placement="H">
<img src="./Process1.png" style="width:80.0%" />
<figcaption>Quarterback history showing Patrick Mahomes’ performance in
his last 16 games.</figcaption>
</figure>

<figure id="fig:defense_history" data-latex-placement="H">
<img src="./Process1Defense.png" style="width:80.0%" />
<figcaption>Defensive history showing the Buffalo Bills’ defensive
performance in their last 8 games.</figcaption>
</figure>

Next, the raw data is normalized to ensure comparability across players
and teams. For instance, Mahomes' passing yards are scaled using the
formula $(331 - 280) / 75 = 0.68$, where 280 is the league average and
75 is the standard deviation. This normalization process is shown in
Figure [7](#fig:normalization){reference-type="ref"
reference="fig:normalization"}.

<figure id="fig:normalization" data-latex-placement="H">
<img src="./ProcessNorm.png" style="width:80.0%" />
<figcaption>Normalization of quarterback statistics to ensure
consistency across the dataset.</figcaption>
</figure>

The normalized data is then passed through LSTM layers, which extract
temporal patterns, such as Mahomes' improvement against blitz-heavy
defenses or the Bills' vulnerabilities against elite quarterbacks. The
LSTM outputs are shown in Figure
[8](#fig:lstm_processing){reference-type="ref"
reference="fig:lstm_processing"}.

<figure id="fig:lstm_processing" data-latex-placement="H">
<img src="./ProcessProcessing.png" style="width:80.0%" />
<figcaption>LSTM outputs for both quarterback and defensive histories,
capturing temporal patterns.</figcaption>
</figure>

Static embeddings for Mahomes and the Bills are added next, representing
unique characteristics like throwing accuracy or defensive tendencies.
These embeddings are shown in Figure
[9](#fig:embeddings){reference-type="ref" reference="fig:embeddings"}.

<figure id="fig:embeddings" data-latex-placement="H">
<img src="./ProcessEmbed.png" style="width:80.0%" />
<figcaption>Embeddings for Patrick Mahomes and the Buffalo Bills,
capturing static player and team traits.</figcaption>
</figure>

The LSTM outputs and embeddings are concatenated into a unified feature
vector, integrating temporal trends and static characteristics. This
concatenated feature vector is visualized in Figure
[10](#fig:concatenated_features){reference-type="ref"
reference="fig:concatenated_features"}.

<figure id="fig:concatenated_features" data-latex-placement="H">
<img src="./ProcessConcate.png" style="width:80.0%" />
<figcaption>Concatenated feature vector combining LSTM outputs and
embeddings for further processing.</figcaption>
</figure>

The combined features are refined through dense layers to model complex
interactions, such as the relationship between Mahomes' offensive
tendencies and the Bills' defensive strengths. The outputs from the
dense layers are shown in Figure
[11](#fig:dense_layers){reference-type="ref"
reference="fig:dense_layers"}.

<figure id="fig:dense_layers" data-latex-placement="H">
<img src="./ProcessDense.png" style="width:80.0%" />
<figcaption>Dense layer outputs, refining concatenated features for
final predictions.</figcaption>
</figure>

Finally, the output layer produces predictions for key metrics,
including passing yards, touchdowns, interceptions, completion
percentage, and yards per attempt. These predictions are visualized in
Figure [12](#fig:output_layer){reference-type="ref"
reference="fig:output_layer"}.

<figure id="fig:output_layer" data-latex-placement="H">
<img src="./ProcessOutput.png" style="width:80.0%" />
<figcaption>Output layer showing predictions for Patrick Mahomes against
the Buffalo Bills.</figcaption>
</figure>

This walkthrough highlights how the LSTM-based model integrates
sequential and static data to generate actionable predictions. However,
while effective, this architecture has limitations in its ability to
dynamically prioritize the most relevant parts of historical sequences
or fully capture complex, long-range interactions between features. To
address these challenges, we developed a transformer-based model, which
builds on the foundation of this architecture while introducing advanced
techniques for temporal modeling and attention.

The new model processes quarterback and defensive play data in a
fundamentally different and more comprehensive manner compared to the
LSTM-based architecture. Instead of limiting the analysis to a fixed
number of games, the transformer-based model aggregates data from up to
the last 2,000 plays, capturing a larger and more detailed dataset that
provides a richer context for predictions. These plays are grouped into
game or week-level segments, and the model applies a dynamic weighting
mechanism to prioritize the most recent performances.

<figure id="fig:transformer_model" data-latex-placement="H">
<img src="./transformermodelexample.png" style="width:80.0%" />
<figcaption> Transformer Model Example: Predicted Week 6 performance for
Patrick Mahomes against the Buffalo Bills.</figcaption>
</figure>

Specifically, the most recent plays are given linear weights ranging
from 1.5 (most recent plays) to 1.0 (older plays), ensuring that recent
trends---such as a quarterback's improvement in form or a defense's
adaptation to new strategies---have the greatest influence on
predictions. For cases where fewer than 2,000 plays are available, the
remaining slots are filled with zero-weighted values, effectively
neutralizing their impact on the model. This approach balances the
importance of historical context with the predictive value of recent
trends, while avoiding bias from incomplete datasets.

By incorporating up to 2,000 plays and dynamically weighting them, the
transformer-based model offers a significant advantage in its ability to
evaluate both short-term and long-term patterns. Combined with its
self-attention mechanisms, this design enables the model to focus on the
most relevant segments of data, ensuring that predictions are accurate,
contextually aware, and robust across diverse scenarios. This
enhancement addresses one of the primary limitations of the LSTM-based
architecture, which treated all historical data equally and was
constrained by fixed-length sequences.

## Improvements <a name="improvements"></a>

To enhance the model's capabilities, several promising improvements are
under consideration. A key direction involves adjusting the neural
network to predict performance metrics for other offensive positions,
including wide receivers, running backs, and tight ends. Expanding the
model to these positions would enable predictions for total yards,
receptions, rushing attempts, and touchdowns, offering a more
comprehensive view of offensive performance and its contributing
factors.

Another important enhancement is the expansion of metrics incorporated
into the model. While the current version provides robust predictions
for quarterback performance, integrating advanced statistics such as
Expected Points Added (EPA), Win Probability Added (WPA), and passer
rating would provide richer contextual insights. These metrics offer a
more nuanced understanding of game dynamics, particularly in
high-pressure or pivotal moments.

Additionally, there is an opportunity to include more detailed context
for opposing defenses. While the model currently considers basic
defensive statistics, incorporating deeper insights---such as defensive
schemes, individual player performance, and coverage tendencies---could
significantly improve the accuracy of predictions, particularly against
complex defensive units. This would allow the model to better assess how
a quarterback might perform against specific defensive alignments or
strategies.

One major factor currently missing is offensive line information.
Offensive line performance has a direct impact on quarterback play,
influencing time in the pocket, sack rates, and overall offensive
efficiency. Incorporating metrics such as pressure rates, sacks allowed,
and offensive line injuries would improve the model's ability to predict
games where protection (or lack thereof) plays a critical role. This
data, combined with defensive context, would enable the model to better
handle scenarios with extreme outcomes, such as games with high sack
totals or exceptional offensive line performance.

Finally, these improvements collectively aim to address a limitation in
the model's current design: its difficulty in predicting "outlier
games", where specific matchups have an outsized impact on player
performance. By integrating advanced metrics, deeper defensive context,
and offensive line data, the model could better identify and account for
games where matchups, injuries, or extreme circumstances lead to
unexpected performances, whether positive or negative. These refinements
would make the model not only more versatile but also more accurate in
capturing the dynamic nature of football gameplay.

## Limitations <a name="limitations"></a>

One major limitation is the absence of weather conditions, such as rain
and snow which are known to significantly impact gameplay. For instance,
precipitation or snow fall can reduce passing efficiency, while colder
temperatures often affect player stamina and ball handling.
Incorporating these factors could make predictions more realistic,
especially for games played in cities prone to adverse weather
conditions during the later part of the season.

Another factor currently missing is the inclusion of home/away dynamics.
Home-field advantage plays a substantial role in determining player and
team performance, influencing crowd noise, travel fatigue, and comfort
levels. Accounting for whether a quarterback is playing at home or on
the road could provide critical context for predictions.

Additionally, the model does not yet integrate metrics related to
formations and receiver quality, which are pivotal to understanding
quarterback performance. Offensive formations, such as shotgun or
play-action setups, significantly influence passing success, while
receiver quality (e.g., route-running ability, separation, and yards
after catch) directly impacts a quarterback's efficiency. Including
these metrics would allow the model to better contextualize quarterback
performance based on the offensive system and surrounding talent.

The model also does not currently extend to different position groups,
limiting its scope to quarterback performance. This narrow focus
excludes insights into how running backs, wide receivers, and tight ends
contribute to overall offensive success. By accounting for other
positions, the model could provide a more holistic view of offensive
dynamics and their effect on quarterbacks.

Furthermore, the model struggles to predict "big games" or outlier
performances by quarterbacks, where they greatly exceed expectations,
often due to extraordinary matchups or circumstances. These rare but
impactful games are difficult to capture without deeper contextual data.
Similarly, the model does not account for blowout potential, where
lopsided games may lead to changes in playcalling, such as resting
starters or shifting to a run-heavy strategy, which could skew
predictions for key metrics like passing yards or touchdowns.

Addressing these limitations in future iterations would significantly
enhance the model's predictive capabilities, providing more contextually
aware and reliable insights for both expected and unexpected scenarios.

## What We Learned <a name="learnings"></a>

Through this project, we developed a deeper understanding of LSTM
architectures and RNNs while enhancing our skills in Python libraries
such as Matplotlib, PyTorch, and Scikit-learn. The importance of data
pre-processing and managing large datasets became especially evident
during this work, as we dealt with one of the most comprehensive
datasets we have ever worked with. Exploratory Data Analysis (EDA) was a
particularly time-intensive process, even more so than in previous
projects, due to the extensive data available through nflfastpy. The
dataset spanned multiple tables, each containing dozens of columns with
various types of play, game, and player data, which posed unique
challenges in understanding and structuring the data.

To tackle this complexity, we created a data dictionary to document the
metadata for each table and column. This helped us understand the
purpose of each feature and allowed us to make informed decisions about
which data to include in our model. Several factors guided our feature
selection process. First, we prioritized relevance to ensure the data
aligned with our project's objectives. Second, we focused on
completeness, favoring columns with minimal missing data since
quarterback statistics are inherently limited by the number of games
played. Finally, we identified categorical attributes and carefully
decided on encoding strategies to make these features usable in our
model. By organizing this metadata in the data dictionary, we
streamlined the feature selection process and established a solid
foundation for building the model.

This project highlighted the challenges and importance of working with
large, complex datasets. Parsing through tables with dozens of
attributes required a structured approach to ensure meaningful insights
could be extracted. The experience emphasized the value of EDA as a
critical step in machine learning workflows, as well as the need for
robust pre-processing techniques to effectively manage data at scale.
Overall, this process provided invaluable learning opportunities and
further strengthened our skills in data analysis, pre-processing, and
machine learning implementation.

# Conclusion <a name="conclusion"></a>

In this project, we developed a machine learning model to predict NFL
quarterback performance, designed to deliver actionable insights for
both fantasy football participants and sports bettors. Initially built
on a Long Short-Term Memory (LSTM) architecture, the model evolved into
a transformer-based design that significantly enhances its ability to
capture complex temporal relationships and contextual dynamics. This
transformation highlights the importance of adapting cutting-edge
techniques to tackle the intricacies of player performance, enabling the
model to provide more precise and robust predictions.

The model integrates play-level data, embeddings, and advanced weighting
mechanisms to account for recent trends and historical context. By
prioritizing recent plays with exponential weighting and leveraging
multi-head attention mechanisms, the transformer architecture excels at
identifying the most relevant patterns in quarterback and defensive
performance. This capability ensures the model is well-suited for
real-world applications, such as optimizing fantasy football lineups and
evaluating player-based prop bets. The growing popularity of these
industries underscores the demand for accurate player performance
forecasts, and this project demonstrates how machine learning can
address that demand.

Despite its strong performance, the model has limitations that highlight
opportunities for further improvement. Currently, the absence of weather
conditions, home/away factors, and detailed receiver metrics constrains
the model's ability to account for all key variables influencing player
performance. Additionally, the model struggles to predict extreme
outlier games or scenarios involving blowouts, where unique matchups or
in-game dynamics lead to unexpected results. Expanding the model to
include advanced metrics like Expected Points Added (EPA), Win
Probability Added (WPA), and offensive line performance would enhance
its predictive power and contextual awareness. Furthermore, adapting the
architecture to predict performance for other offensive positions---such
as wide receivers, running backs, and tight ends---could broaden its
scope and utility, making it a comprehensive tool for analyzing team
dynamics.

The transition from the LSTM-based model to the transformer architecture
underscores the importance of innovation in tackling complex problems in
sports analytics. The transformer model's ability to dynamically
prioritize relevant data and incorporate richer feature sets represents
a significant leap forward in predictive accuracy and flexibility. This
evolution demonstrates how machine learning can bridge the gap between
raw NFL data and actionable insights, empowering users to make informed
decisions in two rapidly growing industries.

Looking ahead, we aim to refine the model further by addressing its
current limitations and incorporating additional features that capture
the nuanced factors influencing NFL games. These enhancements will
ensure the model remains relevant and reliable in dynamic scenarios,
such as late-season games, playoffs, and extreme weather conditions. By
continuing to evolve the architecture and expand its scope, this project
demonstrates the potential of machine learning to revolutionize sports
analytics, empowering fantasy football enthusiasts and sports bettors
with cutting-edge, data-driven tools.

# References <a name="references"></a>

- Chakraborty, I., & Deshmukh, R. (2024, October). *Fantasy Sports Market Size, Share, Competitive Landscape and Trend Analysis Report, by Sports Type, by Platform, by Demographics : Global Opportunity Analysis and Industry Forecast, 2024-2033*. Allied Market Research. [Link](https://www.alliedmarketresearch.com/fantasy-sports-market-A06468)
- Legal Sports Betting. (2024, November). *Legal Sports Betting*. [Link](https://www.legalsportsbetting.com/how-much-money-do-americans-bet-on-sports/)
- Oehy, A. (2024, August). *Fantasy Football financials -- NFL’s fan engagement engine*. Medium (The AO). [Link](https://medium.com/the-ao/fantasy-football-nfls-fan-engagement-engine-c713699859e6)
