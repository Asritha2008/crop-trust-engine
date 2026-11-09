import React from 'react';

export default function InputControl({
  field,
  bounds,
  value,
  onChange,
  error
}) {
  const numericValue = value === '' ? '' : Number(value);

  function handleSliderChange(e) {
    onChange(field, Number(e.target.value));
  }

  function handleNumberChange(e) {
    const val = e.target.value;
    onChange(field, val === '' ? '' : Number(val));
  }

  return (
    <div className={`input-control-card ${error ? 'has-error' : ''}`}>
      <div className="input-control-header">
        <label className="input-control-label" htmlFor={`input-num-${field}`}>
          <span className="field-name">{bounds.label}</span>
          {bounds.unit ? <span className="field-unit">({bounds.unit})</span> : null}
        </label>
        <span className="typical-badge" title="Typical range in training dataset">
          Typical: {bounds.typical}
        </span>
      </div>

      <div className="input-control-body">
        {/* Slider */}
        <div className="slider-wrapper">
          <input
            type="range"
            id={`input-slider-${field}`}
            className="input-range-slider"
            min={bounds.min}
            max={bounds.max}
            step={bounds.step}
            value={numericValue === '' ? bounds.min : numericValue}
            onChange={handleSliderChange}
          />
          <div className="slider-limits">
            <span>{bounds.min}</span>
            <span>{bounds.max}</span>
          </div>
        </div>

        {/* Number Box */}
        <div className="number-box-wrapper">
          <input
            type="number"
            id={`input-num-${field}`}
            className="input-number-field"
            min={bounds.min}
            max={bounds.max}
            step={bounds.step}
            value={value}
            onChange={handleNumberChange}
          />
          {bounds.unit ? <span className="number-unit-suffix">{bounds.unit}</span> : null}
        </div>
      </div>

      {error && <div className="field-error-message">⚠️ {error}</div>}
    </div>
  );
}
