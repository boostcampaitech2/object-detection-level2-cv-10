import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-8s %(levelname)-6s %(message)s",
    datefmt="%m-%d %H:%M",
)
# filename=f'FocalLoss_SGD1e-3_{cur_time_str}.woMaskLoss.log',
# filemode='w')

logger = logging.getLogger('Model')
