document.addEventListener("DOMContentLoaded", function(){
  var options0 = {
    chart: {
      type: 'line'
    },
    series: [{
      name: 'sales',
      data: [30,40,35,50,49,60,70,91,125]
    }],
    xaxis: {
      categories: [1991,1992,1993,1994,1995,1996,1997, 1998,1999]
    }
  }

  var chart0 = new ApexCharts(document.querySelector("#chart0"), options0);

  chart0.render();
});

document.addEventListener("DOMContentLoaded", function(){
  var options1 = {
    series: [44, 55, 13, 43, 22],
    chart: {
      width: 380,
      type: 'pie',
    },
    labels: ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
    responsive: [{
      breakpoint: 480,
      options: {
        chart: {
          width: 200
        },
        legend: {
          position: 'bottom'
        }
      }
    }]
  };
  
  var chart1 = new ApexCharts(document.querySelector("#chart"), options1);
  chart1.render();
});