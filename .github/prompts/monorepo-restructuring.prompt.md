---
name: restructureMonorepo
description: Restructure single-purpose repository into monorepo with shared type definitions
argument-hint: backend-folder-name, frontend-folder-name, shared-folder-name
---

You are an expert at repository restructuring and monorepo architecture. Your task is to help transform a single-purpose repository into a well-organized monorepo that can house multiple related projects (e.g., backend and frontend) with shared type definitions and constants.

## Task Overview

Restructure the current repository into a monorepo with the following goals:
1. Separate existing code into a dedicated backend folder
2. Create a placeholder structure for frontend application
3. Extract shared constants and type definitions into a separate shared folder
4. Maintain git history using `git mv` commands
5. Update all import paths to reference the new structure
6. Create synchronized type definitions for both backend and frontend languages

## Requirements

### Folder Structure
Create the following structure:
```
repository-root/
├── backend-folder/          # Existing backend code moved here
│   ├── [existing folders]
│   └── README.md           # Backend-specific documentation
├── frontend-folder/         # Placeholder for frontend
│   └── README.md           # Frontend documentation
├── shared/                  # Shared code and types
│   ├── constants/          # Shared constants (both languages)
│   └── types/              # Type definitions (both languages)
└── docs/                   # Project documentation
```

### Shared Constants
1. Identify constants duplicated across files
2. Extract to `shared/constants/` in both backend and frontend languages
3. Keep both versions synchronized with clear comments
4. Include validation values, enums, mappings, colors, and thresholds

### Type Definitions
1. Create API contract definitions in `shared/types/`
2. For backend: Use appropriate validation framework (e.g., Pydantic for Python)
3. For frontend: Use TypeScript interfaces matching backend models
4. Ensure type definitions include:
   - Request/response models
   - Enum types
   - Validation constraints (min/max values, required fields)
   - Optional fields with defaults

### Import Path Updates
1. Update all backend files to import from shared folder
2. Add shared folder to module search paths
3. Remove duplicate constant definitions
4. Replace manual JSON construction with type-safe model instances

### Git History Preservation
1. Use `git mv` to move files and preserve history
2. Commit restructuring separately from code changes
3. Create detailed commit messages explaining changes

### Documentation
1. Update root README with monorepo overview
2. Create component-specific README files
3. Document how to import shared constants/types
4. Add migration guide explaining changes
5. Include commands reference for running each component

## Implementation Steps

1. **Analyze Current Structure**: Review existing codebase to identify:
   - Constants to extract
   - Type definitions to create
   - Files to move
   - Import dependencies

2. **Create Folder Structure**: 
   - Create backend, frontend, shared, and docs folders
   - Add .gitkeep files for empty folders

3. **Extract Shared Constants**:
   - Identify duplicate constants across files
   - Create backend language version
   - Create frontend language version
   - Add synchronization comments

4. **Create Type Contracts**:
   - Define API request/response types
   - Add validation constraints
   - Create matching frontend interfaces
   - Include enum types and literals

5. **Move Existing Code**:
   - Use `git mv` to move folders to backend directory
   - Preserve all git history
   - Move documentation files

6. **Update Import Paths**:
   - Add shared folder to module paths
   - Update all imports in backend files
   - Remove duplicate definitions
   - Replace manual response construction with models

7. **Create Documentation**:
   - Update root README
   - Create component READMEs
   - Add migration plan
   - Document shared library usage

8. **Commit Changes**:
   - Stage all changes
   - Create descriptive commit messages
   - Separate restructuring from logic changes

## Output Requirements

Provide:
1. Complete folder structure with all files created
2. Synchronized constant files in both languages
3. Type definition files with validation
4. Updated backend files with new imports
5. Comprehensive documentation
6. Git commands executed
7. Commit messages used

## Best Practices

- Use `git mv` instead of regular `mv` to preserve history
- Keep shared files manually synchronized until automation is added
- Use clear naming conventions (snake_case for backend, camelCase for frontend)
- Add comments indicating files must be kept in sync
- Include validation examples in documentation
- Provide commands for running each component
- Document benefits of the new structure

## Validation

After restructuring, verify:
- All imports resolve correctly
- No duplicate constant definitions remain
- Type definitions match between languages
- Git history preserved for moved files
- Documentation is comprehensive
- All components can run independently
