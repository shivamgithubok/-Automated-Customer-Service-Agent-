import React from 'react';

const TheDynamicInterface = ({ response }) => {
  return (
    <div className="dynamic-interface-container">
      <h2>AI Response:</h2>
      <p>{response ? response : "Waiting for a query..."}</p>
    </div>
  );
};

export default TheDynamicInterface;

      {/* <div style={{ position: 'absolute', width: '50%', height: '50%', backgroundColor: '#3a3838ff', borderLeft: '1px solid #242222ff', padding: '16px', zIndex: '20' }}>
        <TheDynamicInterface response={response} />
      </div> */}