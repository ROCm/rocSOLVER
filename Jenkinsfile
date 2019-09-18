@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

import java.nio.file.Path;

rocSOLVERCI:
{

    def rocsolver = new rocProject('rocSOLVER')
    
    def nodes = new dockerNodes(['internal && gfx900 && ubuntu', 'internal && gfx906 && ubuntu'], rocsolver)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        rocsolver.paths.build_command = platform.jenkinsLabel.contains('centos') ? 'sudo cmake3 -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..' :
                                            'sudo cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc ..'
        
        def getRocBLAS = auxiliary.getLibrary('rocBLAS',platform.jenkinsLabel,'develop',true)
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ${getRocBLAS}
                    sudo mkdir build && cd build
                    export PATH=/opt/rocm/bin:$PATH
                    ${project.paths.build_command}
                    sudo make -j32
                """

        platform.runCommand(this, command)
    }

    def testType = auxiliary.isJobStartedByTimer() ? '*daily_lapack*' : '*checkin_lapack*'
    def testCommand =
    {
        platform, project->

        try
        {
            def getRocBLAS = auxiliary.getLibrary('rocBLAS',platform.jenkinsLabel,'develop',true)
            def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/clients/staging
                    """

            platform.runCommand(this, command)
        }
        finally
        {
        }        
    }

    def packageCommand =
    {
        platform, project->

        def getRocBLAS = auxiliary.getLibrary('rocBLAS',platform.jenkinsLabel,'develop',true)
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build",false,true)  

        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
    }

    buildProject(rocsolver, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

