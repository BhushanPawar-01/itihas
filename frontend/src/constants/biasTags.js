export const BIAS_TAG_META = {
  british_legal:     { label: 'British Legal',     bg: 'bg-blue-100',   text: 'text-blue-800'   },
  british_military:  { label: 'British Military',  bg: 'bg-blue-200',   text: 'text-blue-900'   },
  ina_testimony:     { label: 'INA Testimony',     bg: 'bg-green-100',  text: 'text-green-800'  },
  nationalist_press: { label: 'Nationalist Press', bg: 'bg-orange-100', text: 'text-orange-800' },
  academic:          { label: 'Academic',          bg: 'bg-gray-100',   text: 'text-gray-700'   },
  urdu_press:        { label: 'Urdu Press',        bg: 'bg-purple-100', text: 'text-purple-800' },
  regional_press:    { label: 'Regional Press',    bg: 'bg-yellow-100', text: 'text-yellow-800' },
}

export function getBiasTagMeta(tag) {
  return BIAS_TAG_META[tag] ?? { label: tag, bg: 'bg-gray-100', text: 'text-gray-600' }
}