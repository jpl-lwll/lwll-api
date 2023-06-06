# Copyright (c) 2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pandas as pd
import typing
from lwll_api.classes import metrics
from lwll_api.classes.obj_detection_metrics import mean_average_precision, format_obj_detection_data
import pytest

def get_probabilistic_example() -> typing.Tuple[pd.DataFrame, pd.DataFrame, typing.List[str]]:
    preds = pd.DataFrame([
        {'id': '0.jpg', 'bulldozer': 0.283909, 'helicopter': 0.008925, 'miner': 0.450239, 'sailor': 0.256927},
        {'id': '1.jpg', 'bulldozer': 0.192790, 'helicopter': 0.326889, 'miner': 0.289338, 'sailor': 0.190983},
        {'id': '2.jpg', 'bulldozer': 0.111241, 'helicopter': 0.336317, 'miner': 0.184896, 'sailor': 0.367546},
        {'id': '3.jpg', 'bulldozer': 0.014199, 'helicopter': 0.006181, 'miner': 0.590183, 'sailor': 0.389437},
        {'id': '4.jpg', 'bulldozer': 0.116626, 'helicopter': 0.363193, 'miner': 0.172987, 'sailor': 0.347195},
        {'id': '5.jpg', 'bulldozer': 0.429838, 'helicopter': 0.400465, 'miner': 0.147805, 'sailor': 0.021892},
    ])
    actuals = pd.DataFrame([
        {'id': '0.jpg', 'class': 'miner'},
        {'id': '1.jpg', 'class': 'helicopter'},
        {'id': '2.jpg', 'class': 'bulldozer'},
        {'id': '3.jpg', 'class': 'miner'},
        {'id': '4.jpg', 'class': 'sailor'},
        {'id': '5.jpg', 'class': 'helicopter'},
    ])

    classes = ['bulldozer', 'helicopter', 'miner', 'sailor']
    return preds, actuals, classes


def get_label_example() -> typing.Tuple[typing.List[str], typing.List[str], typing.List[str]]:
    y_true = [
        'bulldozer', 'helicopter', 'helicopter', 'bulldozer', 'sailor',
        'miner', 'miner', 'miner', 'helicopter', 'bulldozer'
    ]

    y_pred = [
        'miner', 'miner', 'miner', 'bulldozer', 'sailor', 'miner',
        'bulldozer', 'sailor', 'miner', 'bulldozer'
    ]

    classes = ['bulldozer', 'helicopter', 'miner', 'sailor']

    return y_true, y_pred, classes


class TestMetrics:

    def test_accuracy(self) -> None:
        a = pd.DataFrame([{'id': 'img1.png', 'label': '2'}, {'id': 'img2.png', 'label': '4'}])
        b = pd.DataFrame([{'id': 'img1.png', 'label': '2'}, {'id': 'img2.png', 'label': '6'}])
        assert 0.5 == metrics.accuracy(a, b)
        assert 1.0 == metrics.accuracy(a, a)

    def test_invalid_one_to_one(self) -> None:
        a = pd.DataFrame([{'id': 'img1.png', 'label': '2'}, {'id': 'img2.png', 'label': '4'}])
        b = pd.DataFrame([{'id': 'img1.png', 'label': '2'}, {'id': 'img3.png', 'label': '6'}])
        c = pd.DataFrame([{'id': 'img1.png', 'label': '2'}, {'id': 'img2.png',
                                                             'label': '6'}, {'id': 'img2.png', 'label': '6'}])
        with pytest.raises(Exception):
            metrics._validate_input_ids_one_to_one(a, b)
        with pytest.raises(Exception):
            metrics._validate_input_ids_one_to_one(a, c)

    def test_mean_average_precision_base(self) -> None:
        ground_truth = [
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img2.png', 'person', 439, 157, 556, 241],
            ['img3.png', 'person', 439, 157, 556, 241],
        ]
        predictions = [
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img2.png', 'person', 0.8, 439, 157, 556, 241],
            ['img3.png', 'person', 0.8, 439, 157, 556, 241],
        ]

        assert mean_average_precision(ground_truth, predictions) == 0.62

    def test_mean_average_precision_no_pred_to_gt(self) -> None:
        ground_truth = [
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img2.png', 'person', 439, 157, 556, 241],
            ['img3.png', 'person', 439, 157, 556, 241],
        ]
        predictions = [
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img2.png', 'cat', 0.8, 439, 157, 556, 241],
        ]
        assert mean_average_precision(ground_truth, predictions) == 0.25

    def test_mean_average_precision_mult_gt_classes(self) -> None:
        ground_truth = [
            ['img1.png', 'cat', 439, 157, 556, 241],
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img2.png', 'person', 439, 157, 556, 241],
            ['img3.png', 'person', 439, 157, 556, 241],
        ]
        predictions = [
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img2.png', 'cat', 0.8, 439, 157, 556, 241],
            ['img3.png', 'person', 0.8, 439, 157, 556, 241]
        ]
        assert mean_average_precision(ground_truth, predictions) == 0.28

    def test_mean_average_precision_conf_threshold(self) -> None:
        ground_truth = [
            ['img1.png', 'cat', 439, 157, 556, 241],
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img2.png', 'person', 439, 157, 556, 241],
            ['img3.png', 'person', 439, 157, 556, 241],
        ]
        predictions = [
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img2.png', 'cat', 0.8, 439, 157, 556, 241],
            ['img3.png', 'person', 0.4, 439, 157, 556, 241]
        ]
        assert mean_average_precision(ground_truth, predictions) == 0.28

    def test_mAP_data_conversion(self) -> None:
        actuals = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'cat'},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
        ])
        preds = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'cat', 'confidence': 0.8},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.4},
        ])

        ground_truth = [
            ['img1.png', 'cat', 439, 157, 556, 241],
            ['img1.png', 'person', 439, 157, 556, 241],
            ['img2.png', 'person', 439, 157, 556, 241],
            ['img3.png', 'person', 439, 157, 556, 241],
        ]
        predictions = [
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img1.png', 'person', 0.8, 439, 157, 556, 241],
            ['img2.png', 'cat', 0.8, 439, 157, 556, 241],
            ['img3.png', 'person', 0.4, 439, 157, 556, 241]
        ]

        out_actuals = format_obj_detection_data(actuals)
        out_preds = format_obj_detection_data(preds)
        assert out_actuals == ground_truth
        assert out_preds == predictions

    def test_mAP(self) -> None:
        actuals = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'cat'},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person'},
        ])
        actuals2 = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'cat', 'confidence': 0.8},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
        ])
        preds = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'cat', 'confidence': 0.8},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.4},
        ])
        assert 0.28 == metrics.mAP(preds, actuals)
        assert 1.0 == metrics.mAP(actuals2, actuals)

    def test_mAP_range(self) -> None:
        actuals = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '438, 153, 556, 241', 'class': 'cat'},
            {'id': 'img1.png', 'bbox': '435, 158, 554, 244', 'class': 'person'},
            {'id': 'img2.png', 'bbox': '436, 157, 558, 241', 'class': 'person'},
            {'id': 'img3.png', 'bbox': '432, 151, 555, 247', 'class': 'person'},
        ])
        actuals2 = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'cat', 'confidence': 0.8},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.7},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.6},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.64},
        ])
        preds = pd.DataFrame([
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.2},
            {'id': 'img1.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.3},
            {'id': 'img2.png', 'bbox': '439, 157, 556, 241', 'class': 'cat', 'confidence': 0.1},
            {'id': 'img3.png', 'bbox': '439, 157, 556, 241', 'class': 'person', 'confidence': 0.8},
        ])

        output = {0.5: 0.33, 0.55: 0.33, 0.6: 0.33, 0.65: 0.33, 0.7: 0.33, 0.75: 0.33, 0.8: 0.33, 0.85: 0.08, 0.9: 0.08, 0.95: 0.00}
        assert metrics.mAP_range(preds, actuals) == output

        output = {0.5: 1.0, 0.55: 1.0, 0.6: 1.0, 0.65: 1.0, 0.7: 1.0, 0.75: 1.0, 0.8: 1.0, 0.85: 0.78, 0.9: 0.78, 0.95: 0.06}
        assert metrics.mAP_range(actuals2, actuals) == output

    def test_bleu_perfect(self) -> None:
        df = pd.DataFrame({"id": [3], "text": ["The dog jumped over the fence"]})
        actuals = pd.DataFrame({"id": [3], "text": ["The dog jumped over the fence"]})
        assert metrics.bleu(df, actuals) == [1, 1, 1, 1]

    def test_bleu_half(self) -> None:
        df = pd.DataFrame({"id": [3, 4], "text": ["The dog jumped over the fence", "Today is Sunday before Christmas"]})
        actuals = pd.DataFrame({"id": [3, 4], "text": ["The dog jumped over the funce",
                                                       "some sentance that is clearly not correct"]})
        assert metrics.bleu(df, actuals) == pytest.approx([0.454774, 0.410512, 0.422685, 0.376446], abs=0.0001)

    def test_bleu_invalid_index(self) -> None:
        df = pd.DataFrame({"id": [3], "text": ["The dog jumped over the fence"]})
        actuals = pd.DataFrame({"id": [6], "text": ["The dog jumped over the fence"]})
        with pytest.raises(Exception):
            metrics.bleu(df, actuals)

    def test_top_5_accuracy(self) -> None:
        preds, actuals, classes = get_probabilistic_example()
        assert metrics.top_5_accuracy(actuals['class'].values, preds) == 1

    def test_roc_auc(self) -> None:
        preds, actuals, classes = get_probabilistic_example()
        assert metrics.roc_auc(actuals['class'].values, preds) == 0.6375

    def test_recall(self) -> None:
        y_true, y_pred, classes = get_label_example()
        assert metrics.recall(y_true, y_pred, labels=classes, average="macro") == 0.5

    def test_weighted_accuracy(self) -> None:
        y_true, y_pred, classes = get_label_example()
        metrics.weighted_accuracy(y_true, y_pred) == 0.5

    def test_precision_at_k(self) -> None:
        preds = pd.DataFrame([
            {'id': '0.jpg', 'bulldozer': 0.284939, 'helicopter': 0.198741, 'miner': 0.543467, 'sailor': 0.599817},
            {'id': '1.jpg', 'bulldozer': 0.985621, 'helicopter': 0.042096, 'miner': 0.476416, 'sailor': 0.655518},
            {'id': '2.jpg', 'bulldozer': 0.921119, 'helicopter': 0.683045, 'miner': 0.233365, 'sailor': 0.817009},
            {'id': '3.jpg', 'bulldozer': 0.002957, 'helicopter': 0.864632, 'miner': 0.819524, 'sailor': 0.878258},
            {'id': '4.jpg', 'bulldozer': 0.362387, 'helicopter': 0.310989, 'miner': 0.887864, 'sailor': 0.408083},
            {'id': '5.jpg', 'bulldozer': 0.026761, 'helicopter': 0.973417, 'miner': 0.227632, 'sailor': 0.180889},
        ])

        actuals = np.array([
            'sailor',
            'bulldozer',
            'bulldozer',
            'miner',
            'miner',
            'helicopter',
        ])

        result = metrics.precision_at_k(actuals, preds, k=2, output='df').values
        expected_output = [['bulldozer', 1.0], ['helicopter', 0.5], ['miner', 1.0], ['sailor', 0.0]]

        for i in range(len(expected_output)):
            for j in range(len(expected_output[i])):
                assert result[i][j] == expected_output[i][j]
