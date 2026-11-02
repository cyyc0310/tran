import numpy as np
import pandas as pd

from tests.test_resource_controls import reference_fixture
from transcif.research import resource_controls as controls
from transcif.research import resource_controls_same_season as strategy


def test_future_source_season_labels_do_not_change_fitted_response():
    f,c,_,power,climate=reference_fixture()
    late=f.iloc[:24].copy();late['clock']+=pd.DateOffset(months=6)
    extended=pd.concat([f,late],ignore_index=True)
    labels=np.vstack([c,c[:24]]);past=np.vstack([power,power[:1]])
    before=strategy.fit(extended,labels,past,climate,allowed_countries=('S',))
    changed=labels.copy();changed[-24:]*=1000.
    after=strategy.fit(extended,changed,past,climate,allowed_countries=('S',))
    daily,_,_=controls.day_inputs(f,climate)
    for learner in ('ridge','rf','hgb'):
        for kind in ('B','W'):
            np.testing.assert_array_equal(controls.predict_amplitudes(before,daily,learner,kind),
                                          controls.predict_amplitudes(after,daily,learner,kind))
    assert before['audit']['provided_source_hours']==96
    assert before['audit']['used_source_hours']==72
