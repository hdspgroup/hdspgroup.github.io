export type PersonLink = {
  label: string;
  href: string;
  icon: string;
};

export type Person = {
  name: string;
  title: string;
  description?: string;
  links?: PersonLink[];
};

export type PeopleCategory = {
  title: string;
  description: string;
  icon: string;
  members: Person[];
};

export const principalInvestigator: Person & { interests: string[]; image: string } = {
  name: 'Professor Henry Arguello Fuentes',
  title:
    'Principal Investigator · Ph.D. in Electrical and Computer Engineering · Associate Professor, Universidad Industrial de Santander',
  description:
    'Professor Henry Arguello leads the HDSP Group, focusing on high-dimensional signal processing, compressed sensing, and computational imaging.',
  image: '~/assets/images/prof-henry-arguello.jpeg',
  interests: [
    'Statistical signal processing',
    'Super-resolution',
    'Inverse problems',
    'Optical imaging',
    'Video processing',
    'Hyperspectral imaging',
    'Compressive sensing',
  ],
  links: [
    {
      label: 'Scholar',
      href: 'https://scholar.google.com/citations?user=R7gjbGIAAAAJ&hl=en',
      icon: 'tabler:school',
    },
  ],
};

export const peopleCategories: PeopleCategory[] = [
  {
    title: 'Professors',
    description: 'Faculty members contributing to the academic and scientific leadership of the group.',
    icon: 'tabler:chalkboard',
    members: [{ name: 'Hans Garcia', title: 'Ph.D. Professor' }],
  },
  {
    title: 'Doctoral Students',
    description: 'Ph.D. researchers advancing core HDSP research lines and multidisciplinary projects.',
    icon: 'tabler:microscope',
    members: [
      { name: 'Paul Goyes', title: 'Ph.D. in Computer Science' },
      { name: 'Jhon Lopez', title: 'Ph.D. in Computer Science' },
      { name: 'Kevin Arias', title: 'Ph.D. in Computer Science' },
      { name: 'Emmanuel Martinez', title: 'Ph.D. in Computer Science' },
      { name: 'Leon Suarez', title: 'Ph.D. in Computer Science' },
      { name: 'Juan Carlos Vega', title: 'Ph.D. in Computer Science' },
      { name: 'Sergio Urrea', title: 'Ph.D. in Engineering' },
      { name: 'Pablo Gomez', title: 'Ph.D. in Engineering' },
      { name: 'Roman Jacome', title: 'Ph.D. in Engineering' },
      {
        name: 'Karen Fonseca',
        title: 'Ph.D. in Engineering',
        links: [
          {
            label: 'Scholar',
            href: 'https://scholar.google.com/citations?user=KQCZTqAAAAAJ&hl=en',
            icon: 'tabler:school',
          },
        ],
      },
    ],
  },
  {
    title: 'Master Students',
    description: 'Graduate students exploring applications across systems, electronics, and geophysics.',
    icon: 'tabler:atom-2',
    members: [
      { name: 'Javier Torres', title: 'M.Sc. in Systems and Computer Engineering' },
      { name: 'Sebastian Ardila', title: 'M.Sc. in Electronic Engineering' },
      { name: 'Ana Mantilla', title: 'M.Sc. in Geophysics' },
    ],
  },
  {
    title: 'Undergraduate Students',
    description: 'Student researchers developing projects, prototypes, and scientific training within HDSP.',
    icon: 'tabler:users-group',
    members: [
      {
        name: 'Laura C. Diaz-Delgado',
        title: 'Computer Science Engineering (8th semester)',
        links: [
          {
            label: 'Scholar',
            href: 'https://scholar.google.com/citations?user=jAfnVpoAAAAJ&hl=en',
            icon: 'tabler:school',
          },
        ],
      },
      { name: 'Julio Gutierrez', title: 'Computer Science Engineering (8th semester)' },
      { name: 'Jose Barrios', title: 'Electronic Engineering (8th semester)' },
      { name: 'Lamar Rivera', title: 'Physics (9th semester)' },
      { name: 'Santiago Rodriguez', title: 'Electronic Engineering (8th semester)' },
      { name: 'Carlos Mogollon', title: 'Electronic Engineering (10th semester)' },
      { name: 'Nohelia Agudelo', title: 'Electronic Engineering (8th semester)' },
      { name: 'Javier Quiroga', title: 'Electronic Engineering (8th semester)' },
      { name: 'Ernesto Vasquez', title: 'Electronic Engineering (4th semester)' },
      { name: 'Deisy Camacho', title: 'Mathematics (8th semester)' },
      { name: 'Daniel Diaz', title: 'Electronic Engineering (6th semester)' },
      { name: 'Juan Diego Cardenas', title: 'Computer Science Engineering (6th semester)' },
    ],
  },
  {
    title: 'Administrative and Professionals',
    description: 'Management and professional support that sustains the group’s projects and operations.',
    icon: 'tabler:briefcase',
    members: [
      { name: 'Ana Gutierrez', title: "Master's in Project Management" },
      { name: 'Marcela Rincon', title: 'Industrial Engineer · Specialist in Strategic Management' },
    ],
  },
  {
    title: 'Collaborators',
    description: 'External researchers and academic partners linked to HDSP projects and scientific exchange.',
    icon: 'tabler:world',
    members: [
      { name: 'Said Pertuz', title: 'Ph.D. Professor' },
      { name: 'Sergio Castillo', title: 'Ph.D. Professor' },
      { name: 'Hoover Rueda', title: 'Ph.D. Professor' },
      { name: 'Luis Gonzalez', title: 'M.Sc. Professor' },
      { name: 'Laura Galvis', title: 'Ph.D. Professor' },
      { name: 'Jorge Bacca', title: 'Ph.D. Professor' },
      { name: 'Alejandra Hernandez', title: 'M.Sc. in Geophysics' },
      { name: 'Ofelia Villarreal', title: 'M.Sc. in Electronic Engineering' },
      { name: 'Romario Gualdron', title: 'M.Sc. in Computer Science Engineering' },
      { name: 'Paula Arguello', title: 'Ph.D. student in Computer Science' },
    ],
  },
];
