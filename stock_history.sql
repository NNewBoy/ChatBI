CREATE TABLE IF NOT EXISTS `stock_history` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `ts_code` VARCHAR(20) NOT NULL COMMENT '股票代码',
  `trade_date` DATE NOT NULL COMMENT '交易日期',
  `open` DECIMAL(12,2) DEFAULT NULL COMMENT '开盘价',
  `high` DECIMAL(12,2) DEFAULT NULL COMMENT '最高价',
  `low` DECIMAL(12,2) DEFAULT NULL COMMENT '最低价',
  `close` DECIMAL(12,2) DEFAULT NULL COMMENT '收盘价',
  `pre_close` DECIMAL(12,2) DEFAULT NULL COMMENT '昨收价',
  `change` DECIMAL(12,2) DEFAULT NULL COMMENT '涨跌额',
  `pct_chg` DECIMAL(10,4) DEFAULT NULL COMMENT '涨跌幅',
  `vol` DECIMAL(16,2) DEFAULT NULL COMMENT '成交量(手)',
  `amount` DECIMAL(20,2) DEFAULT NULL COMMENT '成交额(千元)',
  `stock_name` VARCHAR(20) DEFAULT NULL COMMENT '股票名称',
  PRIMARY KEY (`id`),
  KEY `idx_ts_code` (`ts_code`),
  KEY `idx_trade_date` (`trade_date`),
  KEY `idx_ts_code_date` (`ts_code`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='股票历史行情数据';
