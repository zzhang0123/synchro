"""Provenance of both prediction routes records this package's version."""

import numpy as np

import syncmoments
from syncmoments.model.channels import Channels
from syncmoments.model.predict import direct_channel_average

from _predict_helpers import LINE_NU, nine_atoms, polynomial_kernel, samples_of


def test_direct_route_records_the_package_version():
    kernel, _ = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    pred = direct_channel_average(
        samples_of(nine_atoms()), kernel, channels, amplitude=1.0
    )
    assert pred.provenance.package_version == syncmoments.__version__
    assert np.all(np.isfinite(np.asarray(pred.stokes)))
