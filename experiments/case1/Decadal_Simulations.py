import gcsfs
import jax

import numpy as np
import pickle
import xarray

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

gcs = gcsfs.GCSFileSystem(token='anon')

model_name = 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl'

with gcs.open(f'gs://gresearch/neuralgcm/04_30_2024/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)

# adding this to the code for diagnosing P - E:
new_inputs_to_units_mapping = {
  'u': 'm s**-1',
  'v': 'm s**-1',
  't': 'kelvin',
  'z': 'm**2 s**-2',
  'sim_time': 'dimensionless',
  'tracers': {'specific_humidity': 'dimensionless',
            'specific_cloud_liquid_water_content': 'dimensionless',
            'specific_cloud_ice_water_content': 'dimensionless',
             },
    'diagnostics': {'P_minus_E_rate': 'kg m**-2 s**-1'}

}

new_model_config_str = '\n'.join([
        ckpt['model_config_str'],
        f'DimensionalLearnedPrimitiveToWeatherbenchDecoder.inputs_to_units_mapping = {new_inputs_to_units_mapping}',
        'DimensionalLearnedPrimitiveToWeatherbenchDecoder.diagnostics_module = @NodalModelDiagnosticsDecoder',
        'StochasticPhysicsParameterizationStep.diagnostics_module = @PrecipitationMinusEvaporationDiagnostics',
        'PrecipitationMinusEvaporationDiagnostics.method = "rate"',
        'PrecipitationMinusEvaporationDiagnostics.moisture_species =  ("specific_humidity", "specific_cloud_liquid_water_content", "specific_cloud_ice_water_content")',])
ckpt['model_config_str'] = new_model_config_str

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

# for forcing we use the whole period from 1980 to 2023
start_time = '1980-01-01T12:00:00.000000000'
end_time = '2023-12-31'
data_inner_steps = 24

# for climate simulations I don't see a reason for shifting the forcings by 24 hours.
sliced_era5_forcing = (
    full_era5
    [model.forcing_variables]
    # .pipe(
    #     xarray_utils.selective_temporal_shift,
    #     variables=model.forcing_variables,
    #     time_shift='24 hours',
    # )
    .sel(time=slice(start_time, end_time, data_inner_steps))
    .compute()
)

# For regridding ERA5 data to the model resolution.
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

# regridding the forcing first.
era5_forcing = xarray_utils.regrid(sliced_era5_forcing, regridder)
era5_forcing = xarray_utils.fill_nan_with_nearest(era5_forcing)

# path to save the runs
path = '/glade/derecho/scratch/pahlavan/ai-models/ngcm/climate/'

# 37 different initialization dates (spaced 10 days apart)

for t in range(37):
    init_date = np.datetime64('1980-01-01T00:00:00.000000000') + t * np.timedelta64(10,'D')

    # for initial condition
    start_time = init_date
    end_time = init_date
    data_inner_steps = 24  # process every 24th hour
    
    sliced_era5_input = (
        full_era5
        [model.input_variables].sel(time=slice(start_time, end_time, data_inner_steps))
        .compute()
    )

    era5_input = xarray_utils.regrid(sliced_era5_input, regridder)
    era5_input = xarray_utils.fill_nan_with_nearest(era5_input)
    
    t0 = np.datetime64(init_date)
    # initialize model state
    inputs = model.inputs_from_xarray(era5_input.isel(time=0))
    forcing_initial = model.forcings_from_xarray(era5_forcing.isel(time=0))
    rng_key = jax.random.key(42)  # optional for deterministic models
    initial_state = model.encode(inputs, forcing_initial, rng_key)

    dt = np.timedelta64(6, 'h')  # save outputs four times per day
    steps = 50 * 24 // 6   # forecast 50 days at a time
    all_forcings = model.forcings_from_xarray(era5_forcing)  # update forcings
    times = t0 + (np.arange(1, steps+1) * dt)  # time axis in hours
    
    state, outputs = model.unroll(initial_state, all_forcings, steps=steps, timedelta=dt, start_with_input=False)
    outputs_ds = model.data_to_xarray(outputs, times=times)
    outputs_ds_rechunked = outputs_ds.chunk({'time':-1, 'latitude':-1, 'longitude':-1})
    outputs_ds_rechunked.to_zarr(path + str(init_date)[:10] + '.zarr')

    # 320 runs each for 50 days is ~43 years long.
    for d in range(1, 320):
        times = times[-1] + (np.arange(1, steps+1) * dt)
        state, outputs = model.unroll(state, all_forcings, steps=steps, timedelta=dt, start_with_input=False)
        outputs_ds = model.data_to_xarray(outputs, times=times)
        # save outputs to disk
        outputs_ds_rechunked = outputs_ds.chunk({'time':-1, 'latitude':-1, 'longitude':-1})
        outputs_ds_rechunked.to_zarr(path + str(init_date)[:10] + '.zarr', append_dim='time')
