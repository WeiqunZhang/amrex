import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    "Introduction",
    {
      "type": "category",
      "label": "Getting Started",
      "link": {
        "type": "doc",
        "id": "GettingStarted_Chapter"
      },
      "items": [
        "GettingStarted"
      ]
    },
    {
      "type": "category",
      "label": "Building AMReX",
      "link": {
        "type": "doc",
        "id": "BuildingAMReX_Chapter"
      },
      "items": [
        "BuildingAMReX"
      ]
    },
    {
      "type": "category",
      "label": "Basics",
      "link": {
        "type": "doc",
        "id": "Basics_Chapter"
      },
      "items": [
        "Basics"
      ]
    },
    {
      "type": "category",
      "label": "Gridding and Load Balancing",
      "link": {
        "type": "doc",
        "id": "ManagingGridHierarchy_Chapter"
      },
      "items": [
        "GridCreation",
        "DualGrid",
        "LoadBalancing"
      ]
    },
    {
      "type": "category",
      "label": "AmrCore Source Code",
      "link": {
        "type": "doc",
        "id": "AmrCore_Chapter"
      },
      "items": [
        "AmrCore"
      ]
    },
    {
      "type": "category",
      "label": "Amr Source Code",
      "link": {
        "type": "doc",
        "id": "AmrLevel_Chapter"
      },
      "items": [
        "AmrLevel"
      ]
    },
    "ForkJoin",
    {
      "type": "category",
      "label": "I/O (Plotfile, Checkpoint)",
      "link": {
        "type": "doc",
        "id": "IO_Chapter"
      },
      "items": [
        "IO"
      ]
    },
    {
      "type": "category",
      "label": "Linear Solvers",
      "link": {
        "type": "doc",
        "id": "LinearSolvers_Chapter"
      },
      "items": [
        "LinearSolvers"
      ]
    },
    {
      "type": "category",
      "label": "Particles",
      "link": {
        "type": "doc",
        "id": "Particle_Chapter"
      },
      "items": [
        "Particle"
      ]
    },
    {
      "type": "category",
      "label": "Fortran Interface",
      "link": {
        "type": "doc",
        "id": "Fortran_Chapter"
      },
      "items": [
        "Fortran"
      ]
    },
    "Python_Chapter",
    {
      "type": "category",
      "label": "Embedded Boundaries",
      "link": {
        "type": "doc",
        "id": "EB_Chapter"
      },
      "items": [
        "EB"
      ]
    },
    {
      "type": "category",
      "label": "Discrete Fourier Transform",
      "link": {
        "type": "doc",
        "id": "FFT_Chapter"
      },
      "items": [
        "FFT"
      ]
    },
    "TimeIntegration_Chapter",
    {
      "type": "category",
      "label": "GPU",
      "link": {
        "type": "doc",
        "id": "GPU_Chapter"
      },
      "items": [
        "GPU"
      ]
    },
    {
      "type": "category",
      "label": "Visualization",
      "link": {
        "type": "doc",
        "id": "Visualization_Chapter"
      },
      "items": [
        "Visualization"
      ]
    },
    {
      "type": "category",
      "label": "Post-Processing",
      "link": {
        "type": "doc",
        "id": "Post_Processing_Chapter"
      },
      "items": [
        "Post_Processing"
      ]
    },
    "Debugging",
    "RuntimeParameters",
    {
      "type": "category",
      "label": "AMReX-based Profiling Tools",
      "link": {
        "type": "doc",
        "id": "AMReX_Profiling_Tools_Chapter"
      },
      "items": [
        "AMReX_Profiling_Tools"
      ]
    },
    {
      "type": "category",
      "label": "External Profiling Tools",
      "link": {
        "type": "doc",
        "id": "External_Profiling_Tools_Chapter"
      },
      "items": [
        "External_Profiling_Tools"
      ]
    },
    {
      "type": "category",
      "label": "External Frameworks",
      "link": {
        "type": "doc",
        "id": "External_Frameworks_Chapter"
      },
      "items": [
        "SUNDIALS_top"
      ]
    },
    {
      "type": "category",
      "label": "Regression Testing",
      "link": {
        "type": "doc",
        "id": "Regression_Testing_Chapter"
      },
      "items": [
        "Testing"
      ]
    },
    "Faq",
    "Governance"
  ],
};

export default sidebars;
